use pyo3::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;
use rayon::prelude::*;
use rand::prelude::*;
use regex::Regex;
use std::sync::Arc;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Recipe {
    pub title: Option<String>,
    pub ingredients: Option<Vec<String>>,
    pub instructions: Option<Vec<String>>,
    pub cuisine: Option<String>,
    pub cooking_time: Option<String>,
    pub servings: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub ingredient_count: usize,
    pub instruction_steps: usize,
    pub cooking_time_mentioned: bool,
    pub measurements_present: bool,
    pub cooking_methods_count: usize,
    pub complexity_score: f32,
    pub b2b_features_count: usize,
}

#[derive(Debug, Clone)]
pub struct BusinessScenario {
    pub volume_range: (i32, i32),
    pub cost_target: String,
    pub skill_level: String,
    pub keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransformedRecipe {
    pub input: String,
    pub output: String,
    pub format: String,
    pub business_scenario: Option<String>,
    pub volume: Option<i32>,
    pub original_recipe: String,
    pub quality_metrics: QualityMetrics,
}

#[pyclass]
pub struct PyDatasetTransformer {
    business_scenarios: HashMap<String, BusinessScenario>,
    cooking_methods: Vec<String>,
    measurement_regex: Vec<Regex>,
    rng: StdRng,
}

#[pymethods]
impl PyDatasetTransformer {
    #[new]
    pub fn new() -> PyResult<Self> {
        let mut business_scenarios = HashMap::new();
        
        // Restaurant Fast Casual
        business_scenarios.insert("restaurant_fast_casual".to_string(), BusinessScenario {
            volume_range: (100, 500),
            cost_target: "Budget".to_string(),
            skill_level: "Line Cook".to_string(),
            keywords: vec![
                "burger".to_string(), "sandwich".to_string(), "salad".to_string(), 
                "wrap".to_string(), "bowl".to_string(), "quick".to_string(), "fast".to_string()
            ],
        });
        
        // Restaurant Fine Dining
        business_scenarios.insert("restaurant_fine_dining".to_string(), BusinessScenario {
            volume_range: (30, 100),
            cost_target: "Premium".to_string(),
            skill_level: "Professional".to_string(),
            keywords: vec![
                "salmon".to_string(), "beef".to_string(), "lamb".to_string(),
                "duck".to_string(), "lobster".to_string(), "truffle".to_string(), "wine".to_string()
            ],
        });
        
        // Catering Corporate
        business_scenarios.insert("catering_corporate".to_string(), BusinessScenario {
            volume_range: (100, 1000),
            cost_target: "Mid-range".to_string(),
            skill_level: "Professional".to_string(),
            keywords: vec![
                "chicken".to_string(), "pasta".to_string(), "rice".to_string(),
                "vegetables".to_string(), "soup".to_string(), "buffet".to_string()
            ],
        });
        
        // Meal Kit Family
        business_scenarios.insert("meal_kit_family".to_string(), BusinessScenario {
            volume_range: (10000, 50000),
            cost_target: "Mid-range".to_string(),
            skill_level: "Home Cook".to_string(),
            keywords: vec![
                "easy".to_string(), "quick".to_string(), "family".to_string(),
                "kid".to_string(), "simple".to_string(), "30".to_string(), "minute".to_string()
            ],
        });
        
        // Institutional School
        business_scenarios.insert("institutional_school".to_string(), BusinessScenario {
            volume_range: (500, 2000),
            cost_target: "Budget".to_string(),
            skill_level: "Institutional".to_string(),
            keywords: vec![
                "healthy".to_string(), "nutritious".to_string(), "kid-friendly".to_string(),
                "lunch".to_string(), "cafeteria".to_string()
            ],
        });
        
        let cooking_methods = vec![
            "bake", "roast", "grill", "fry", "saute", "boil", "steam", "braise",
            "simmer", "broil", "poach", "blanch", "sear", "caramelize"
        ].into_iter().map(|s| s.to_string()).collect();
        
        // Compile measurement regexes
        let measurement_patterns = vec![
            r"\d+\s*(cup|tablespoon|teaspoon|pound|ounce|gram|liter|ml|tbsp|tsp|lb|oz|g)",
            r"\d+/\d+\s*(cup|tablespoon|teaspoon)",
            r"\d+\.\d+\s*(cup|tablespoon|teaspoon|pound|ounce)",
        ];
        
        let measurement_regex: Result<Vec<Regex>, _> = measurement_patterns
            .into_iter()
            .map(|pattern| Regex::new(pattern))
            .collect();
        
        let measurement_regex = measurement_regex
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("Regex error: {}", e)))?;
        
        Ok(PyDatasetTransformer {
            business_scenarios,
            cooking_methods,
            measurement_regex,
            rng: StdRng::from_entropy(),
        })
    }
    
    /// Transform dataset with parallel processing optimized for Ryzen 9 3900X
    pub fn transform_dataset_parallel(
        &mut self,
        recipes_json: &str,
        b2b_percentage: f64,
        max_recipes: Option<usize>,
    ) -> PyResult<String> {
        // Parse JSON
        let recipes: Vec<Value> = serde_json::from_str(recipes_json)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON parse error: {}", e)))?;
        
        println!("🦀 Rust Dataset Transformer");
        println!("📊 Total recipes loaded: {}", recipes.len());
        
        // Limit dataset size if specified
        let recipes: Vec<Value> = if let Some(max) = max_recipes {
            let mut rng = StdRng::from_entropy();
            let sample_size = std::cmp::min(max, recipes.len());
            recipes.choose_multiple(&mut rng, sample_size).cloned().collect()
        } else {
            recipes
        };
        
        println!("📊 Processing {} recipes", recipes.len());
        
        // Calculate B2B split
        let num_b2b = (recipes.len() as f64 * b2b_percentage) as usize;
        println!("🏢 B2B transformations: {}", num_b2b);
        println!("📝 Regular format: {}", recipes.len() - num_b2b);
        
        // Split recipes randomly
        let mut rng = StdRng::from_entropy();
        let mut recipe_indices: Vec<usize> = (0..recipes.len()).collect();
        recipe_indices.shuffle(&mut rng);
        
        let b2b_indices: Vec<usize> = recipe_indices.into_iter().take(num_b2b).collect();
        let b2b_set: std::collections::HashSet<usize> = b2b_indices.into_iter().collect();
        
        // Prepare scenarios for parallel processing
        let scenarios = Arc::new(self.business_scenarios.clone());
        let cooking_methods = Arc::new(self.cooking_methods.clone());
        let measurement_regex = Arc::new(self.measurement_regex.clone());
        
        // Parallel processing with Rayon
        let start_time = std::time::Instant::now();
        
        let transformed_recipes: Vec<TransformedRecipe> = recipes
            .par_iter()
            .enumerate()
            .map(|(index, recipe_value)| {
                let recipe = self.parse_recipe_value(recipe_value);
                
                if b2b_set.contains(&index) {
                    // B2B transformation
                    self.transform_to_b2b_parallel(
                        &recipe,
                        &scenarios,
                        &cooking_methods,
                        &measurement_regex,
                    )
                } else {
                    // Regular transformation
                    self.transform_to_regular_parallel(&recipe, &cooking_methods, &measurement_regex)
                }
            })
            .collect();
        
        let processing_time = start_time.elapsed();
        
        println!("⚡ Processing complete: {:.2}s", processing_time.as_secs_f64());
        println!("🚀 Speed: {:.1} recipes/sec", recipes.len() as f64 / processing_time.as_secs_f64());
        
        // Validate quality
        let validation_results = self.validate_quality(&transformed_recipes);
        println!("\n🔍 QUALITY VALIDATION:");
        println!("✅ Avg B2B quality: {:.1}/100", validation_results.get("avg_b2b_quality").unwrap_or(&0.0));
        println!("📈 Avg regular quality: {:.1}/100", validation_results.get("avg_regular_quality").unwrap_or(&0.0));
        println!("🏷️  Avg B2B features: {:.1}", validation_results.get("avg_b2b_features").unwrap_or(&0.0));
        
        // Convert to JSON
        let result_json = serde_json::to_string(&transformed_recipes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(format!("JSON serialize error: {}", e)))?;
        
        Ok(result_json)
    }
    
    /// Benchmark transformation performance
    pub fn benchmark_performance(&mut self, sample_size: usize) -> PyResult<HashMap<String, f64>> {
        println!("🏎️  RUST TRANSFORMATION BENCHMARK");
        println!("🦀 Testing {} recipes on Ryzen 9 3900X", sample_size);
        
        // Create test data
        let test_recipes: Vec<Value> = (0..sample_size)
            .map(|i| serde_json::json!({
                "title": format!("Test Recipe {}", i),
                "ingredients": vec![
                    "2 cups flour",
                    "1 cup sugar", 
                    "3 eggs",
                    "1/2 cup butter",
                    "1 tsp vanilla"
                ],
                "instructions": vec![
                    "Mix dry ingredients",
                    "Add wet ingredients", 
                    "Bake at 350F for 30 minutes"
                ]
            }))
            .collect();
        
        let test_json = serde_json::to_string(&test_recipes).unwrap();
        
        // Benchmark different B2B percentages
        let mut results = HashMap::new();
        
        for b2b_pct in vec![0.0, 0.2, 0.5, 1.0] {
            let start = std::time::Instant::now();
            
            let _transformed = self.transform_dataset_parallel(&test_json, b2b_pct, None)?;
            
            let duration = start.elapsed().as_secs_f64();
            let speed = sample_size as f64 / duration;
            
            println!("{}% B2B: {:.2}s ({:.1} recipes/sec)", b2b_pct * 100.0, duration, speed);
            results.insert(format!("speed_{}pct_b2b", (b2b_pct * 100.0) as i32), speed);
        }
        
        Ok(results)
    }
}

impl PyDatasetTransformer {
    fn parse_recipe_value(&self, value: &Value) -> Recipe {
        Recipe {
            title: value.get("title").and_then(|v| v.as_str()).map(|s| s.to_string()),
            ingredients: value.get("ingredients").and_then(|v| {
                if let Some(arr) = v.as_array() {
                    Some(arr.iter().filter_map(|item| item.as_str().map(|s| s.to_string())).collect())
                } else if let Some(s) = v.as_str() {
                    Some(vec![s.to_string()])
                } else {
                    None
                }
            }),
            instructions: value.get("instructions").and_then(|v| {
                if let Some(arr) = v.as_array() {
                    Some(arr.iter().filter_map(|item| item.as_str().map(|s| s.to_string())).collect())
                } else if let Some(s) = v.as_str() {
                    Some(vec![s.to_string()])
                } else {
                    None
                }
            }),
            cuisine: value.get("cuisine").and_then(|v| v.as_str()).map(|s| s.to_string()),
            cooking_time: value.get("cooking_time").and_then(|v| v.as_str()).map(|s| s.to_string()),
            servings: value.get("servings").and_then(|v| v.as_i64()).map(|i| i as i32),
        }
    }
    
    fn classify_recipe_to_business(&self, recipe: &Recipe) -> String {
        let recipe_text = self.get_recipe_text(recipe).to_lowercase();
        let mut scenario_scores = HashMap::new();
        
        for (scenario_name, scenario) in &self.business_scenarios {
            let mut score = 0;
            
            // Keyword matching
            for keyword in &scenario.keywords {
                if recipe_text.contains(keyword) {
                    score += if keyword.len() > 4 { 2 } else { 1 };
                }
            }
            
            // Complexity matching
            let ingredient_count = recipe.ingredients.as_ref().map(|i| i.len()).unwrap_or(0);
            match scenario.skill_level.as_str() {
                "Home Cook" if ingredient_count <= 8 => score += 2,
                "Professional" if ingredient_count >= 6 => score += 2,
                "Line Cook" if ingredient_count >= 4 && ingredient_count <= 10 => score += 1,
                _ => {}
            }
            
            // Time consideration
            if recipe_text.contains("time") || recipe_text.contains("minute") {
                if scenario.skill_level == "Home Cook" || scenario.skill_level == "Line Cook" {
                    score += 1;
                }
            }
            
            scenario_scores.insert(scenario_name.clone(), score);
        }
        
        // Return best match or default
        scenario_scores
            .iter()
            .max_by_key(|(_, &score)| score)
            .map(|(name, _)| name.clone())
            .unwrap_or_else(|| "restaurant_fast_casual".to_string())
    }
    
    fn get_recipe_text(&self, recipe: &Recipe) -> String {
        let mut text_parts = Vec::new();
        
        if let Some(title) = &recipe.title {
            text_parts.push(title.clone());
        }
        
        if let Some(ingredients) = &recipe.ingredients {
            text_parts.extend(ingredients.clone());
        }
        
        if let Some(instructions) = &recipe.instructions {
            text_parts.extend(instructions.clone());
        }
        
        text_parts.join(" ")
    }
    
    fn calculate_quality_metrics(
        &self,
        recipe: &Recipe,
        cooking_methods: &[String],
        measurement_regex: &[Regex],
    ) -> QualityMetrics {
        let recipe_text = self.get_recipe_text(recipe).to_lowercase();
        
        let ingredient_count = recipe.ingredients.as_ref().map(|i| i.len()).unwrap_or(0);
        
        let instruction_steps = recipe.instructions.as_ref()
            .map(|instructions| instructions.len())
            .unwrap_or(0);
        
        let cooking_time_mentioned = ["minute", "hour", "time", "cook for", "bake for"]
            .iter()
            .any(|&word| recipe_text.contains(word));
        
        let measurements_present = measurement_regex
            .iter()
            .any(|regex| regex.is_match(&recipe_text));
        
        let cooking_methods_count = cooking_methods
            .iter()
            .filter(|&method| recipe_text.contains(method))
            .count();
        
        let complexity_score = std::cmp::min(100, 
            ingredient_count * 5 +
            instruction_steps * 3 +
            cooking_methods_count * 10 +
            if cooking_time_mentioned { 20 } else { 0 } +
            if measurements_present { 15 } else { 0 }
        ) as f32;
        
        QualityMetrics {
            ingredient_count,
            instruction_steps,
            cooking_time_mentioned,
            measurements_present,
            cooking_methods_count,
            complexity_score,
            b2b_features_count: 0,
        }
    }
    
    fn transform_to_b2b_parallel(
        &self,
        recipe: &Recipe,
        scenarios: &Arc<HashMap<String, BusinessScenario>>,
        cooking_methods: &Arc<Vec<String>>,
        measurement_regex: &Arc<Vec<Regex>>,
    ) -> TransformedRecipe {
        let scenario_name = self.classify_recipe_to_business(recipe);
        let scenario = scenarios.get(&scenario_name).unwrap();
        
        let mut rng = StdRng::from_entropy();
        let volume = rng.gen_range(scenario.volume_range.0..=scenario.volume_range.1);
        
        let b2b_prompt = self.create_b2b_prompt(recipe, scenario, volume);
        let b2b_output = self.create_b2b_output(recipe, scenario, volume);
        
        let mut quality_metrics = self.calculate_quality_metrics(recipe, cooking_methods, measurement_regex);
        quality_metrics.b2b_features_count = self.count_b2b_features(&b2b_output);
        
        TransformedRecipe {
            input: b2b_prompt,
            output: b2b_output,
            format: "b2b_enterprise".to_string(),
            business_scenario: Some(scenario_name),
            volume: Some(volume),
            original_recipe: recipe.title.clone().unwrap_or_else(|| "Unknown".to_string()),
            quality_metrics,
        }
    }
    
    fn transform_to_regular_parallel(
        &self,
        recipe: &Recipe,
        cooking_methods: &Arc<Vec<String>>,
        measurement_regex: &Arc<Vec<Regex>>,
    ) -> TransformedRecipe {
        let ingredients_text = recipe.ingredients.as_ref()
            .map(|ingredients| ingredients.join(", "))
            .unwrap_or_else(|| "No ingredients listed".to_string());
        
        let instructions_text = recipe.instructions.as_ref()
            .map(|instructions| instructions.join(" "))
            .unwrap_or_else(|| "No instructions provided".to_string());
        
        let input = format!("Create {}", recipe.title.as_ref().unwrap_or(&"a recipe".to_string()));
        let output = format!("Ingredients: {}\nInstructions: {}", ingredients_text, instructions_text);
        
        let quality_metrics = self.calculate_quality_metrics(recipe, cooking_methods, measurement_regex);
        
        TransformedRecipe {
            input,
            output,
            format: "simple".to_string(),
            business_scenario: None,
            volume: None,
            original_recipe: recipe.title.clone().unwrap_or_else(|| "Unknown".to_string()),
            quality_metrics,
        }
    }
    
    fn create_b2b_prompt(&self, recipe: &Recipe, scenario: &BusinessScenario, volume: i32) -> String {
        let default_title = "recipe".to_string();
        let title = recipe.title.as_ref().unwrap_or(&default_title);
        
        format!(
            r#"[BUSINESS_REQUEST]
[BUSINESS_TYPE]Commercial Kitchen[/BUSINESS_TYPE]
[SERVICE_STYLE]{} Service[/SERVICE_STYLE]
[VOLUME]{} servings[/VOLUME]
[COST_TARGET]{}[/COST_TARGET]
[SKILL_LEVEL]{}[/SKILL_LEVEL]
[MEAL_STRUCTURE]Complete Meal[/MEAL_STRUCTURE]

Create a commercial version of {} optimized for business food service.

[REQUIREMENTS]
- Food cost control and portion consistency
- Equipment efficiency and workflow optimization
- Food safety and temperature control compliance
- Scalable preparation methods for volume production
- Standardized procedures for staff training
[/REQUIREMENTS]
[/BUSINESS_REQUEST]

Generate enterprise recipe:"#,
            scenario.cost_target, volume, scenario.cost_target, scenario.skill_level, title
        )
    }
    
    fn create_b2b_output(&self, recipe: &Recipe, scenario: &BusinessScenario, volume: i32) -> String {
        let default_title = "Commercial Recipe".to_string();
        let title = recipe.title.as_ref().unwrap_or(&default_title);
        let ingredients = recipe.ingredients.as_ref().cloned().unwrap_or_else(|| vec!["Ingredients not specified".to_string()]);
        let instructions = recipe.instructions.as_ref().cloned().unwrap_or_else(|| vec!["Instructions not provided".to_string()]);
        
        let scaled_ingredients = self.scale_ingredients(&ingredients, volume);
        
        let mut output = format!(
            r#"[RECIPE_START]
[TITLE_START]{} (Commercial - {} servings)[TITLE_END]
[BUSINESS_INFO_START]
[COST_TARGET]{}[/COST_TARGET]
[SKILL_LEVEL]{}[/SKILL_LEVEL]
[VOLUME]{} servings[/VOLUME]
[PREP_TIME]Optimized for volume production[/PREP_TIME]
[/BUSINESS_INFO_END]

[EQUIPMENT_START]
- Commercial-grade equipment required
- Food safety temperature monitoring
- Portion control tools for consistency
[EQUIPMENT_END]

[INGREDIENTS_START]"#,
            title, volume, scenario.cost_target, scenario.skill_level, volume
        );
        
        for ingredient in scaled_ingredients {
            output.push_str(&format!("\n[INGREDIENT]{}[/INGREDIENT]", ingredient));
        }
        
        output.push_str("\n[INGREDIENTS_END]\n\n[INSTRUCTIONS_START]");
        
        for (i, instruction) in instructions.iter().enumerate() {
            output.push_str(&format!(
                "\n[STEP]{}[/STEP][TECHNIQUE]Commercial Method[/TECHNIQUE] {}",
                i + 1, instruction
            ));
        }
        
        output.push_str(&format!(
            r#"
[INSTRUCTIONS_END]

[BUSINESS_NOTES_START]
- Cost target: {}
- Staff skill level: {}
- Volume optimized: {} servings
- Food safety compliance required
- Portion control for consistency
[BUSINESS_NOTES_END]
[RECIPE_END]"#,
            scenario.cost_target, scenario.skill_level, volume
        ));
        
        output
    }
    
    fn scale_ingredients(&self, ingredients: &[String], target_volume: i32) -> Vec<String> {
        let base_servings = 4.0;
        let scale_factor = target_volume as f64 / base_servings;
        
        let number_regex = Regex::new(r"(\d+(?:\.\d+)?)").unwrap();
        
        ingredients
            .iter()
            .map(|ingredient| {
                number_regex.replace_all(ingredient, |caps: &regex::Captures| {
                    let num: f64 = caps[1].parse().unwrap_or(1.0);
                    let scaled = num * scale_factor;
                    format!("{:.2}", scaled).trim_end_matches('0').trim_end_matches('.').to_string()
                }).to_string()
            })
            .collect()
    }
    
    fn count_b2b_features(&self, b2b_output: &str) -> usize {
        let b2b_tokens = [
            "[BUSINESS_TYPE]", "[SERVICE_STYLE]", "[VOLUME]", "[COST_TARGET]",
            "[SKILL_LEVEL]", "[EQUIPMENT_START]", "[BUSINESS_NOTES_START]",
            "[TECHNIQUE]", "[STEP]"
        ];
        
        b2b_tokens
            .iter()
            .filter(|&token| b2b_output.contains(token))
            .count()
    }
    
    fn validate_quality(&self, transformed_recipes: &[TransformedRecipe]) -> HashMap<String, f64> {
        let b2b_recipes: Vec<_> = transformed_recipes
            .iter()
            .filter(|r| r.format == "b2b_enterprise")
            .collect();
        
        let regular_recipes: Vec<_> = transformed_recipes
            .iter()
            .filter(|r| r.format == "simple")
            .collect();
        
        let avg_b2b_quality = if !b2b_recipes.is_empty() {
            b2b_recipes.iter().map(|r| r.quality_metrics.complexity_score as f64).sum::<f64>() / b2b_recipes.len() as f64
        } else {
            0.0
        };
        
        let avg_regular_quality = if !regular_recipes.is_empty() {
            regular_recipes.iter().map(|r| r.quality_metrics.complexity_score as f64).sum::<f64>() / regular_recipes.len() as f64
        } else {
            0.0
        };
        
        let avg_b2b_features = if !b2b_recipes.is_empty() {
            b2b_recipes.iter().map(|r| r.quality_metrics.b2b_features_count as f64).sum::<f64>() / b2b_recipes.len() as f64
        } else {
            0.0
        };
        
        let mut results = HashMap::new();
        results.insert("avg_b2b_quality".to_string(), avg_b2b_quality);
        results.insert("avg_regular_quality".to_string(), avg_regular_quality);
        results.insert("avg_b2b_features".to_string(), avg_b2b_features);
        
        results
    }
}