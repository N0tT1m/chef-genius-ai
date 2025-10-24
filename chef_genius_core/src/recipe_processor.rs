use pyo3::prelude::*;
use crate::models::*;
use std::collections::HashMap;
use regex::Regex;
use std::sync::Arc;
use parking_lot::RwLock;

pub struct RecipeProcessor {
    ingredient_db: Arc<RwLock<HashMap<String, IngredientInfo>>>,
    substitution_rules: Arc<RwLock<Vec<SubstitutionRule>>>,
    dietary_filters: Arc<RwLock<HashMap<String, Vec<String>>>>,
    stats: Arc<RwLock<ProcessorStats>>,
}

#[derive(Debug, Clone)]
struct IngredientInfo {
    name: String,
    category: String,
    aliases: Vec<String>,
    nutrition_per_100g: HashMap<String, f32>,
    allergens: Vec<String>,
    dietary_tags: Vec<String>,
}

#[derive(Debug, Clone)]
struct SubstitutionRule {
    from_ingredient: String,
    to_ingredient: String,
    ratio: f32,
    conditions: Vec<String>,
    dietary_compatibility: Vec<String>,
}

#[derive(Debug, Default)]
struct ProcessorStats {
    recipes_processed: u64,
    ingredients_parsed: u64,
    substitutions_applied: u64,
    avg_processing_time_ms: f64,
}

#[pyclass]
pub struct PyRecipeProcessor {
    processor: Arc<RwLock<RecipeProcessor>>,
}

#[pymethods]
impl PyRecipeProcessor {
    #[new]
    fn new() -> PyResult<Self> {
        let processor = RecipeProcessor::new_internal()?;
        Ok(Self {
            processor: Arc::new(RwLock::new(processor)),
        })
    }

    /// Parse ingredients from text
    fn parse_ingredients(&self, py: Python, text: String) -> PyResult<Vec<PyObject>> {
        let processor = self.processor.clone();
        
        py.allow_threads(|| {
            let processor = processor.read();
            let ingredients = processor.parse_ingredients_internal(&text)?;
            
            Python::with_gil(|py| {
                let result: Vec<PyObject> = ingredients
                    .into_iter()
                    .map(|ing| {
                        let dict = pyo3::types::PyDict::new(py);
                        dict.set_item("name", ing.name).unwrap();
                        dict.set_item("quantity", ing.quantity).unwrap();
                        dict.set_item("unit", ing.unit).unwrap();
                        dict.set_item("preparation", ing.preparation).unwrap();
                        dict.to_object(py)
                    })
                    .collect();
                Ok(result)
            })
        })
    }

    /// Clean and standardize recipe text
    fn clean_recipe_text(&self, py: Python, text: String) -> PyResult<String> {
        let processor = self.processor.clone();
        
        py.allow_threads(|| {
            let processor = processor.read();
            processor.clean_recipe_text_internal(&text)
        })
    }

    /// Find ingredient substitutions
    fn find_substitutions(&self, py: Python, ingredient: String, dietary_restrictions: Option<Vec<String>>) -> PyResult<Vec<PyObject>> {
        let processor = self.processor.clone();
        
        py.allow_threads(|| {
            let processor = processor.read();
            let substitutions = processor.find_substitutions_internal(&ingredient, dietary_restrictions.as_ref())?;
            
            Python::with_gil(|py| {
                let result: Vec<PyObject> = substitutions
                    .into_iter()
                    .map(|sub| {
                        let dict = pyo3::types::PyDict::new(py);
                        dict.set_item("substitute", sub.substitute).unwrap();
                        dict.set_item("ratio", sub.ratio).unwrap();
                        dict.set_item("notes", sub.notes).unwrap();
                        dict.set_item("confidence", sub.confidence).unwrap();
                        dict.to_object(py)
                    })
                    .collect();
                Ok(result)
            })
        })
    }

    /// Extract cooking instructions steps
    fn extract_cooking_steps(&self, py: Python, instructions: String) -> PyResult<Vec<String>> {
        let processor = self.processor.clone();
        
        py.allow_threads(|| {
            let processor = processor.read();
            processor.extract_cooking_steps_internal(&instructions)
        })
    }

    /// Estimate cooking time from instructions
    fn estimate_cooking_time(&self, py: Python, instructions: Vec<String>) -> PyResult<PyObject> {
        let processor = self.processor.clone();
        
        py.allow_threads(|| {
            let processor = processor.read();
            let time_estimate = processor.estimate_cooking_time_internal(&instructions)?;
            
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("prep_time_minutes", time_estimate.prep_time_minutes)?;
                dict.set_item("cook_time_minutes", time_estimate.cook_time_minutes)?;
                dict.set_item("total_time_minutes", time_estimate.total_time_minutes)?;
                dict.set_item("confidence", time_estimate.confidence)?;
                Ok(dict.to_object(py))
            })
        })
    }

    /// Validate recipe completeness
    fn validate_recipe(&self, py: Python, recipe: PyRecipe) -> PyResult<PyObject> {
        let processor = self.processor.clone();
        
        py.allow_threads(|| {
            let processor = processor.read();
            let validation = processor.validate_recipe_internal(&recipe)?;
            
            Python::with_gil(|py| {
                let dict = pyo3::types::PyDict::new(py);
                dict.set_item("is_valid", validation.is_valid)?;
                dict.set_item("score", validation.score)?;
                dict.set_item("issues", validation.issues)?;
                dict.set_item("suggestions", validation.suggestions)?;
                Ok(dict.to_object(py))
            })
        })
    }

    /// Get processing statistics
    fn get_stats(&self) -> PyResult<PyObject> {
        let processor = self.processor.read();
        let stats = processor.stats.read();
        
        Python::with_gil(|py| {
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("recipes_processed", stats.recipes_processed)?;
            dict.set_item("ingredients_parsed", stats.ingredients_parsed)?;
            dict.set_item("substitutions_applied", stats.substitutions_applied)?;
            dict.set_item("avg_processing_time_ms", stats.avg_processing_time_ms)?;
            Ok(dict.to_object(py))
        })
    }
}

#[derive(Debug)]
struct ParsedIngredient {
    name: String,
    quantity: Option<f32>,
    unit: Option<String>,
    preparation: Option<String>,
}

#[derive(Debug)]
struct SubstitutionSuggestion {
    substitute: String,
    ratio: f32,
    notes: String,
    confidence: f32,
}

#[derive(Debug)]
struct TimeEstimate {
    prep_time_minutes: u32,
    cook_time_minutes: u32,
    total_time_minutes: u32,
    confidence: f32,
}

#[derive(Debug)]
struct RecipeValidation {
    is_valid: bool,
    score: f32,
    issues: Vec<String>,
    suggestions: Vec<String>,
}

impl RecipeProcessor {
    fn new_internal() -> PyResult<Self> {
        let mut processor = Self {
            ingredient_db: Arc::new(RwLock::new(HashMap::new())),
            substitution_rules: Arc::new(RwLock::new(Vec::new())),
            dietary_filters: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(ProcessorStats::default())),
        };
        
        processor.initialize_ingredient_db();
        processor.initialize_substitution_rules();
        processor.initialize_dietary_filters();
        
        Ok(processor)
    }

    fn initialize_ingredient_db(&mut self) {
        let mut db = self.ingredient_db.write();
        
        // Add common ingredients with nutrition info
        let ingredients = vec![
            ("chicken breast", "protein", vec!["chicken", "poultry"], "calories:165,protein:31,fat:3.6,carbs:0"),
            ("rice", "grains", vec!["white rice", "jasmine rice"], "calories:130,protein:2.7,fat:0.3,carbs:28"),
            ("broccoli", "vegetables", vec!["broccoli florets"], "calories:34,protein:2.8,fat:0.4,carbs:7"),
            ("salmon", "protein", vec!["salmon fillet", "fish"], "calories:208,protein:22,fat:12,carbs:0"),
            ("eggs", "protein", vec!["egg", "chicken eggs"], "calories:155,protein:13,fat:11,carbs:1.1"),
            ("olive oil", "fats", vec!["extra virgin olive oil"], "calories:884,protein:0,fat:100,carbs:0"),
            ("onion", "vegetables", vec!["yellow onion", "white onion"], "calories:40,protein:1.1,fat:0.1,carbs:9.3"),
            ("garlic", "vegetables", vec!["garlic cloves"], "calories:149,protein:6.4,fat:0.5,carbs:33"),
            ("tomato", "vegetables", vec!["tomatoes", "fresh tomato"], "calories:18,protein:0.9,fat:0.2,carbs:3.9"),
            ("pasta", "grains", vec!["spaghetti", "penne", "linguine"], "calories:131,protein:5,fat:1.1,carbs:25"),
        ];

        for (name, category, aliases, nutrition_str) in ingredients {
            let nutrition_map: HashMap<String, f32> = nutrition_str
                .split(',')
                .filter_map(|pair| {
                    let parts: Vec<&str> = pair.split(':').collect();
                    if parts.len() == 2 {
                        if let Ok(value) = parts[1].parse::<f32>() {
                            return Some((parts[0].to_string(), value));
                        }
                    }
                    None
                })
                .collect();

            let ingredient_info = IngredientInfo {
                name: name.to_string(),
                category: category.to_string(),
                aliases: aliases.into_iter().map(|s| s.to_string()).collect(),
                nutrition_per_100g: nutrition_map,
                allergens: Vec::new(),
                dietary_tags: Vec::new(),
            };

            db.insert(name.to_string(), ingredient_info);
        }
    }

    fn initialize_substitution_rules(&mut self) {
        let mut rules = self.substitution_rules.write();
        
        let substitutions = vec![
            ("butter", "olive oil", 0.75, vec![], vec!["vegan"]),
            ("heavy cream", "coconut cream", 1.0, vec![], vec!["vegan", "dairy-free"]),
            ("chicken breast", "tofu", 1.0, vec![], vec!["vegan", "vegetarian"]),
            ("beef", "mushrooms", 0.8, vec![], vec!["vegan", "vegetarian"]),
            ("wheat flour", "almond flour", 0.75, vec![], vec!["gluten-free", "keto"]),
            ("sugar", "stevia", 0.1, vec![], vec!["keto", "diabetic"]),
            ("milk", "almond milk", 1.0, vec![], vec!["vegan", "dairy-free"]),
            ("eggs", "flax eggs", 1.0, vec!["baking"], vec!["vegan"]),
            ("cheese", "nutritional yeast", 0.25, vec![], vec!["vegan", "dairy-free"]),
            ("honey", "maple syrup", 1.0, vec![], vec!["vegan"]),
        ];

        for (from, to, ratio, conditions, dietary) in substitutions {
            let rule = SubstitutionRule {
                from_ingredient: from.to_string(),
                to_ingredient: to.to_string(),
                ratio,
                conditions: conditions.into_iter().map(|s| s.to_string()).collect(),
                dietary_compatibility: dietary.into_iter().map(|s| s.to_string()).collect(),
            };
            rules.push(rule);
        }
    }

    fn initialize_dietary_filters(&mut self) {
        let mut filters = self.dietary_filters.write();
        
        filters.insert("vegan".to_string(), vec![
            "meat".to_string(), "poultry".to_string(), "fish".to_string(), 
            "dairy".to_string(), "eggs".to_string(), "honey".to_string()
        ]);
        
        filters.insert("vegetarian".to_string(), vec![
            "meat".to_string(), "poultry".to_string(), "fish".to_string()
        ]);
        
        filters.insert("gluten-free".to_string(), vec![
            "wheat".to_string(), "barley".to_string(), "rye".to_string(), 
            "flour".to_string(), "bread".to_string(), "pasta".to_string()
        ]);
        
        filters.insert("dairy-free".to_string(), vec![
            "milk".to_string(), "cheese".to_string(), "butter".to_string(), 
            "cream".to_string(), "yogurt".to_string()
        ]);
    }

    fn parse_ingredients_internal(&self, text: &str) -> PyResult<Vec<ParsedIngredient>> {
        let start_time = std::time::Instant::now();
        
        // Regular expressions for parsing ingredient lines
        let ingredient_regex = Regex::new(
            r"(?i)^(?:[-*•]?\s*)?(?P<quantity>\d+(?:\.\d+)?(?:/\d+)?(?:\s*[-–]\s*\d+(?:\.\d+)?)?)\s*(?P<unit>(?:cups?|tbsp|tsp|lbs?|oz|g|kg|ml|l|cloves?|pieces?|slices?)\.?)\s+(?P<name>.*?)(?:\s*,\s*(?P<prep>.*))?$"
        ).unwrap();
        
        let simple_regex = Regex::new(
            r"(?i)^(?:[-*•]?\s*)?(?P<name>[^,]+?)(?:\s*,\s*(?P<prep>.*))?$"
        ).unwrap();

        let mut ingredients = Vec::new();
        
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }

            // Try detailed parsing first
            if let Some(caps) = ingredient_regex.captures(line) {
                let quantity = caps.name("quantity")
                    .and_then(|m| m.as_str().parse::<f32>().ok());
                let unit = caps.name("unit")
                    .map(|m| m.as_str().to_string());
                let name = caps.name("name")
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let preparation = caps.name("prep")
                    .map(|m| m.as_str().trim().to_string());

                ingredients.push(ParsedIngredient {
                    name,
                    quantity,
                    unit,
                    preparation,
                });
            }
            // Fall back to simple parsing
            else if let Some(caps) = simple_regex.captures(line) {
                let name = caps.name("name")
                    .map(|m| m.as_str().trim().to_string())
                    .unwrap_or_else(|| "unknown".to_string());
                let preparation = caps.name("prep")
                    .map(|m| m.as_str().trim().to_string());

                ingredients.push(ParsedIngredient {
                    name,
                    quantity: None,
                    unit: None,
                    preparation,
                });
            }
        }

        // Update stats
        {
            let mut stats = self.stats.write();
            stats.ingredients_parsed += ingredients.len() as u64;
            let processing_time = start_time.elapsed().as_millis() as f64;
            stats.avg_processing_time_ms = (stats.avg_processing_time_ms * (stats.recipes_processed as f64) + processing_time) / (stats.recipes_processed + 1) as f64;
            stats.recipes_processed += 1;
        }

        Ok(ingredients)
    }

    fn clean_recipe_text_internal(&self, text: &str) -> PyResult<String> {
        let mut cleaned = text.to_string();
        
        // Remove extra whitespace
        cleaned = Regex::new(r"\s+").unwrap().replace_all(&cleaned, " ").to_string();
        
        // Fix common typos and formatting
        cleaned = cleaned.replace("°F", "°F");
        cleaned = cleaned.replace("°C", "°C");
        
        // Standardize fractions
        cleaned = cleaned.replace("1/2", "½");
        cleaned = cleaned.replace("1/3", "⅓");
        cleaned = cleaned.replace("1/4", "¼");
        cleaned = cleaned.replace("3/4", "¾");
        
        // Remove excessive punctuation
        cleaned = Regex::new(r"[.]{3,}").unwrap().replace_all(&cleaned, "...").to_string();
        cleaned = Regex::new(r"[!]{2,}").unwrap().replace_all(&cleaned, "!").to_string();
        
        Ok(cleaned.trim().to_string())
    }

    fn find_substitutions_internal(&self, ingredient: &str, dietary_restrictions: Option<&Vec<String>>) -> PyResult<Vec<SubstitutionSuggestion>> {
        let rules = self.substitution_rules.read();
        let mut suggestions = Vec::new();
        
        let ingredient_lower = ingredient.to_lowercase();
        
        for rule in rules.iter() {
            if rule.from_ingredient.to_lowercase() == ingredient_lower || 
               ingredient_lower.contains(&rule.from_ingredient.to_lowercase()) {
                
                // Check if substitution meets dietary requirements
                let meets_dietary_reqs = if let Some(restrictions) = dietary_restrictions {
                    restrictions.iter().any(|restriction| {
                        rule.dietary_compatibility.contains(restriction)
                    })
                } else {
                    true
                };
                
                if meets_dietary_reqs {
                    let confidence = if rule.from_ingredient.to_lowercase() == ingredient_lower {
                        0.9
                    } else {
                        0.7
                    };
                    
                    let notes = if rule.ratio != 1.0 {
                        format!("Use {} times the amount", rule.ratio)
                    } else {
                        "Direct substitution".to_string()
                    };
                    
                    suggestions.push(SubstitutionSuggestion {
                        substitute: rule.to_ingredient.clone(),
                        ratio: rule.ratio,
                        notes,
                        confidence,
                    });
                }
            }
        }
        
        // Update stats
        {
            let mut stats = self.stats.write();
            stats.substitutions_applied += suggestions.len() as u64;
        }
        
        Ok(suggestions)
    }

    fn extract_cooking_steps_internal(&self, instructions: &str) -> PyResult<Vec<String>> {
        let mut steps = Vec::new();
        
        // Split by common step indicators
        let step_regex = Regex::new(r"(?i)(?:^|\n)\s*(?:\d+\.?\s*|step\s*\d+:?\s*|•\s*|[-*]\s*)(.+?)(?=\n\s*(?:\d+\.?\s*|step\s*\d+:?\s*|•\s*|[-*]\s*)|$)").unwrap();
        
        for cap in step_regex.captures_iter(instructions) {
            if let Some(step) = cap.get(1) {
                let step_text = step.as_str().trim();
                if !step_text.is_empty() && step_text.len() > 10 {
                    steps.push(step_text.to_string());
                }
            }
        }
        
        // If no numbered steps found, split by sentences
        if steps.is_empty() {
            for sentence in instructions.split('.') {
                let sentence = sentence.trim();
                if sentence.len() > 15 {
                    steps.push(format!("{}.", sentence));
                }
            }
        }
        
        Ok(steps)
    }

    fn estimate_cooking_time_internal(&self, instructions: &[String]) -> PyResult<TimeEstimate> {
        let mut prep_time = 0u32;
        let mut cook_time = 0u32;
        
        // Time extraction regex
        let time_regex = Regex::new(r"(?i)(\d+)(?:\s*[-–]\s*(\d+))?\s*(minutes?|mins?|hours?|hrs?)").unwrap();
        
        for instruction in instructions {
            let instruction_lower = instruction.to_lowercase();
            
            // Extract time mentions
            for cap in time_regex.captures_iter(instruction) {
                if let Ok(time) = cap[1].parse::<u32>() {
                    let unit = &cap[3];
                    let time_in_minutes = if unit.starts_with("hour") || unit.starts_with("hr") {
                        time * 60
                    } else {
                        time
                    };
                    
                    // Categorize as prep or cook time based on context
                    if instruction_lower.contains("chop") || 
                       instruction_lower.contains("dice") ||
                       instruction_lower.contains("slice") ||
                       instruction_lower.contains("prepare") ||
                       instruction_lower.contains("mix") {
                        prep_time = prep_time.max(time_in_minutes);
                    } else if instruction_lower.contains("cook") ||
                              instruction_lower.contains("bake") ||
                              instruction_lower.contains("simmer") ||
                              instruction_lower.contains("boil") ||
                              instruction_lower.contains("fry") {
                        cook_time = cook_time.max(time_in_minutes);
                    }
                }
            }
            
            // Estimate based on cooking methods if no explicit times
            if instruction_lower.contains("bake") && cook_time == 0 {
                cook_time = 30; // Default baking time
            } else if instruction_lower.contains("simmer") && cook_time == 0 {
                cook_time = 20; // Default simmering time
            } else if instruction_lower.contains("sauté") && cook_time == 0 {
                cook_time = 10; // Default sautéing time
            }
        }
        
        // Default estimates if nothing found
        if prep_time == 0 {
            prep_time = 15; // Default prep time
        }
        if cook_time == 0 {
            cook_time = 25; // Default cook time
        }
        
        let confidence = if instructions.iter().any(|i| time_regex.is_match(i)) {
            0.8
        } else {
            0.5
        };
        
        Ok(TimeEstimate {
            prep_time_minutes: prep_time,
            cook_time_minutes: cook_time,
            total_time_minutes: prep_time + cook_time,
            confidence,
        })
    }

    fn validate_recipe_internal(&self, recipe: &PyRecipe) -> PyResult<RecipeValidation> {
        let mut score = 0.0f32;
        let mut issues = Vec::new();
        let mut suggestions = Vec::new();
        
        // Check title
        if recipe.title.is_empty() {
            issues.push("Recipe has no title".to_string());
        } else if recipe.title.len() < 3 {
            issues.push("Recipe title is too short".to_string());
            suggestions.push("Consider adding a more descriptive title".to_string());
        } else {
            score += 15.0;
        }
        
        // Check ingredients
        if recipe.ingredients.is_empty() {
            issues.push("Recipe has no ingredients".to_string());
        } else if recipe.ingredients.len() < 2 {
            issues.push("Recipe has very few ingredients".to_string());
            suggestions.push("Consider adding more ingredients for complexity".to_string());
        } else {
            score += 25.0;
            if recipe.ingredients.len() >= 5 {
                score += 10.0; // Bonus for detailed ingredient list
            }
        }
        
        // Check instructions
        if recipe.instructions.is_empty() {
            issues.push("Recipe has no instructions".to_string());
        } else if recipe.instructions.len() < 2 {
            issues.push("Recipe has very few instructions".to_string());
            suggestions.push("Consider breaking down the cooking process into more steps".to_string());
        } else {
            score += 30.0;
            if recipe.instructions.len() >= 4 {
                score += 10.0; // Bonus for detailed instructions
            }
        }
        
        // Check optional fields
        if recipe.cooking_time.is_some() {
            score += 5.0;
        } else {
            suggestions.push("Consider adding cooking time information".to_string());
        }
        
        if recipe.servings.is_some() {
            score += 5.0;
        } else {
            suggestions.push("Consider adding serving size information".to_string());
        }
        
        if recipe.difficulty.is_some() {
            score += 5.0;
        } else {
            suggestions.push("Consider adding difficulty level".to_string());
        }
        
        if recipe.cuisine_type.is_some() {
            score += 5.0;
        } else {
            suggestions.push("Consider specifying the cuisine type".to_string());
        }
        
        let is_valid = issues.is_empty() && score >= 50.0;
        
        Ok(RecipeValidation {
            is_valid,
            score,
            issues,
            suggestions,
        })
    }
}