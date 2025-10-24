mod jsonl_converter;

use anyhow::{Context, Result};
use clap::{Arg, Command};
use comfy_table::{modifiers::UTF8_ROUND_CORNERS, presets::UTF8_FULL, Table};
use csv::ReaderBuilder;
use indexmap::IndexMap;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidatedRecipe {
    pub id: String,
    pub title: String,
    pub ingredients: Vec<String>,
    pub instructions: Vec<String>,
    pub metadata: RecipeMetadata,
    pub quality_score: f64,
    pub mistral_input: String,
    pub mistral_output: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecipeMetadata {
    pub cuisine: Option<String>,
    pub course: Option<String>,
    pub cooking_time: Option<String>,
    pub servings: Option<String>,
    pub difficulty: Option<String>,
    pub source: Option<String>,
    pub original_format: String,
}

#[derive(Debug, Clone)]
pub struct QualityMetrics {
    pub has_title: bool,
    pub has_ingredients: bool,
    pub has_instructions: bool,
    pub ingredient_count: usize,
    pub instruction_count: usize,
    pub avg_instruction_length: f64,
    pub contains_measurements: bool,
    pub contains_cooking_terms: bool,
    pub total_word_count: usize,
    pub is_duplicate: bool,
    pub has_junk_data: bool,
}

#[derive(Debug)]
pub struct ValidationStats {
    pub total_processed: usize,
    pub valid_recipes: usize,
    pub duplicates_found: usize,
    pub parsing_errors: usize,
    pub quality_failures: usize,
    pub junk_data_found: usize,
    pub format_distribution: HashMap<String, usize>,
    pub quality_distribution: HashMap<String, usize>,
}

pub struct RecipeValidator {
    seen_hashes: Arc<Mutex<HashMap<String, usize>>>,
    cooking_terms: Vec<String>,
    measurement_regex: Regex,
    list_regex: Regex,
    json_regex: Regex,
    junk_data_regex: Regex,
}

impl RecipeValidator {
    pub fn new() -> Self {
        let cooking_terms = vec![
            "bake", "boil", "simmer", "fry", "sauté", "grill", "roast", "steam",
            "mix", "stir", "whisk", "fold", "chop", "dice", "mince", "slice",
            "preheat", "season", "marinate", "garnish", "serve", "cook", "heat",
            "add", "combine", "blend", "pour", "place", "cover", "remove",
        ].into_iter().map(|s| s.to_string()).collect();

        let measurement_regex = Regex::new(
            r"(?i)\b\d+(\.\d+)?\s*(cup|cups|tsp|tbsp|teaspoon|tablespoon|oz|ounce|lb|pound|g|gram|kg|ml|liter|quart|pint)\b"
        ).unwrap();

        let list_regex = Regex::new(r"^\s*\[.*\]\s*$").unwrap();
        let json_regex = Regex::new(r#"^\s*\[".*"\]\s*$"#).unwrap();
        
        // Pattern to detect junk data: excessive punctuation, emoticons, inappropriate text, non-recipe content
        let junk_data_regex = Regex::new(r"(?i)(!!{3,}|:\-?\)|:D|XD|[xX]{3,}|lol|omg|wtf|haha|rofl|lmao|yummy!!+|delicious!!+|awesome!!+|amazing!!+|best\s+.*\s+ever!!+|chow!!+|no\s+peek(?:ing)?!!+|enjoy!!+|yum!!+|mmm{3,}|<3|www\.|http|\.com|\.net|\.org|email|@|#|hashtag|follow|like|share|subscribe|click|link|website|blog|facebook|twitter|instagram|youtube|tiktok|snapchat|pinterest|reddit|tumblr|linkedin|copyright|©|\(c\)|all\s+rights\s+reserved|disclaimer|terms\s+of\s+use|privacy\s+policy|contact\s+us|about\s+us|home\s+page|site\s+map|search|login|register|sign\s+up|sign\s+in|logout|account|profile|settings|preferences|dashboard|admin|moderator|member|guest|user|visitor|customer|client|subscriber|follower|fan|friend|buddy|pal|mate|dude|guy|girl|lady|man|woman|person|people|folks|everyone|everybody|somebody|someone|nobody|anyone|anywhere|everywhere|nowhere|somewhere|anytime|sometime|never|always|usually|often|sometimes|rarely|seldom|hardly|barely|almost|nearly|quite|very|really|extremely|incredibly|absolutely|totally|completely|entirely|fully|partially|somewhat|rather|fairly|pretty|kind\s+of|sort\s+of|type\s+of|kind\s+like|sort\s+like|type\s+like|similar\s+to|different\s+from|same\s+as|equal\s+to|identical\s+to|equivalent\s+to|comparable\s+to|related\s+to|connected\s+to|linked\s+to|associated\s+with|affiliated\s+with|partnered\s+with|sponsored\s+by|endorsed\s+by|recommended\s+by|approved\s+by|certified\s+by|verified\s+by|tested\s+by|reviewed\s+by|rated\s+by|ranked\s+by|listed\s+by|featured\s+by|highlighted\s+by|mentioned\s+by|cited\s+by|referenced\s+by|quoted\s+by|discussed\s+by|talked\s+about|written\s+about|posted\s+about|shared\s+about|commented\s+on|replied\s+to|responded\s+to|answered\s+to|reacted\s+to|liked\s+by|loved\s+by|favorited\s+by|bookmarked\s+by|saved\s+by|downloaded\s+by|uploaded\s+by|created\s+by|made\s+by|produced\s+by|manufactured\s+by|developed\s+by|designed\s+by|built\s+by|constructed\s+by|assembled\s+by)").unwrap();

        Self {
            seen_hashes: Arc::new(Mutex::new(HashMap::new())),
            cooking_terms,
            measurement_regex,
            list_regex,
            json_regex,
            junk_data_regex,
        }
    }

    pub fn validate_csv(&self, file_path: &str, format: &str) -> Result<(Vec<ValidatedRecipe>, ValidationStats)> {
        println!("🔍 Validating dataset: {}", file_path);
        println!("📋 Expected format: {}", format);

        let file = File::open(file_path)
            .with_context(|| format!("Failed to open file: {}", file_path))?;

        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .from_reader(file);

        let headers: Vec<String> = reader.headers()?.iter().map(|h| h.to_string()).collect();
        println!("📊 Headers found: {:?}", headers);

        let records: Vec<csv::StringRecord> = reader.records().collect::<Result<Vec<_>, _>>()?;
        let total_count = records.len();

        println!("📈 Processing {} records...", total_count);

        let pb = ProgressBar::new(total_count as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, {eta})")
                .unwrap()
                .progress_chars("#>-"),
        );

        let stats = Arc::new(Mutex::new(ValidationStats {
            total_processed: 0,
            valid_recipes: 0,
            duplicates_found: 0,
            parsing_errors: 0,
            quality_failures: 0,
            junk_data_found: 0,
            format_distribution: HashMap::new(),
            quality_distribution: HashMap::new(),
        }));

        let valid_recipes: Vec<ValidatedRecipe> = records
            .par_iter()
            .enumerate()
            .filter_map(|(idx, record)| {
                pb.inc(1);
                
                let result = match format {
                    "recipe_nlg" => self.parse_recipe_nlg(record, &headers, idx),
                    "food_com_raw" => self.parse_food_com_raw(record, &headers, idx),
                    "food_com_dataset" => self.parse_food_com_dataset(record, &headers, idx),
                    "indian_food" => self.parse_indian_food(record, &headers, idx),
                    "recipe_images" => self.parse_recipe_images(record, &headers, idx),
                    "recipe_box" => self.parse_recipe_box(record, &headers, idx),
                    _ => {
                        eprintln!("❌ Unknown format: {}", format);
                        return None;
                    }
                };

                let mut stats_lock = stats.lock().unwrap();
                stats_lock.total_processed += 1;

                match result {
                    Ok(recipe) => {
                        let quality = self.calculate_quality_metrics(&recipe);
                        let quality_score = self.calculate_quality_score(&quality);
                        
                        // Track junk data
                        if quality.has_junk_data {
                            stats_lock.junk_data_found += 1;
                        }
                        
                        if quality_score >= 0.5 {
                            stats_lock.valid_recipes += 1;
                            
                            // Track quality distribution
                            let quality_tier = if quality_score >= 0.8 { "High" }
                            else if quality_score >= 0.6 { "Medium" } 
                            else { "Low" };
                            *stats_lock.quality_distribution.entry(quality_tier.to_string()).or_insert(0) += 1;

                            // Track format
                            *stats_lock.format_distribution.entry(recipe.metadata.original_format.clone()).or_insert(0) += 1;

                            Some(recipe)
                        } else {
                            stats_lock.quality_failures += 1;
                            None
                        }
                    }
                    Err(_) => {
                        stats_lock.parsing_errors += 1;
                        None
                    }
                }
            })
            .collect();

        pb.finish_with_message("✅ Validation complete");

        let final_stats = Arc::try_unwrap(stats).unwrap().into_inner().unwrap();
        Ok((valid_recipes, final_stats))
    }

    fn parse_recipe_nlg(&self, record: &csv::StringRecord, headers: &[String], idx: usize) -> Result<ValidatedRecipe> {
        let id = format!("recipe_nlg_{}", idx);
        let title = self.get_field(record, headers, "title")?;
        
        let ingredients_raw = self.get_field(record, headers, "ingredients")?;
        let ingredients = self.safe_parse_list(&ingredients_raw, "ingredients")?;
        
        let directions_raw = self.get_field(record, headers, "directions")?;
        let instructions = self.safe_parse_list(&directions_raw, "directions")?;
        
        let source_url = record.get(headers.iter().position(|h| h == "link").unwrap_or(0))
            .unwrap_or("")
            .to_string();

        let metadata = RecipeMetadata {
            cuisine: None,
            course: None,
            cooking_time: None,
            servings: None,
            difficulty: None,
            source: if source_url.is_empty() { None } else { Some(source_url) },
            original_format: "RecipeNLG".to_string(),
        };

        let (mistral_input, mistral_output) = self.generate_mistral_format(&title, &ingredients, &instructions, &metadata);
        let quality_metrics = self.calculate_quality_metrics_from_parts(&title, &ingredients, &instructions);
        let quality_score = self.calculate_quality_score(&quality_metrics);

        Ok(ValidatedRecipe {
            id,
            title,
            ingredients,
            instructions,
            metadata,
            quality_score,
            mistral_input,
            mistral_output,
        })
    }

    fn parse_food_com_raw(&self, record: &csv::StringRecord, headers: &[String], idx: usize) -> Result<ValidatedRecipe> {
        let id = format!("food_com_{}", self.get_field(record, headers, "id").unwrap_or_else(|_| idx.to_string()));
        let title = self.get_field(record, headers, "name")?;
        
        let ingredients_raw = self.get_field(record, headers, "ingredients")?;
        let ingredients = self.safe_parse_list(&ingredients_raw, "ingredients")?;
        
        let steps_raw = self.get_field(record, headers, "steps")?;
        let instructions = self.safe_parse_list(&steps_raw, "steps")?;
        
        let minutes = self.get_field(record, headers, "minutes").ok();
        let cooking_time = minutes.map(|m| format!("{} minutes", m));

        let metadata = RecipeMetadata {
            cuisine: None,
            course: None,
            cooking_time,
            servings: None,
            difficulty: None,
            source: Some("Food.com".to_string()),
            original_format: "Food.com RAW".to_string(),
        };

        let (mistral_input, mistral_output) = self.generate_mistral_format(&title, &ingredients, &instructions, &metadata);
        let quality_metrics = self.calculate_quality_metrics_from_parts(&title, &ingredients, &instructions);
        let quality_score = self.calculate_quality_score(&quality_metrics);

        Ok(ValidatedRecipe {
            id,
            title,
            ingredients,
            instructions,
            metadata,
            quality_score,
            mistral_input,
            mistral_output,
        })
    }

    fn parse_food_com_dataset(&self, record: &csv::StringRecord, headers: &[String], idx: usize) -> Result<ValidatedRecipe> {
        // This format has no instructions, so we'll skip it or create minimal entries
        Err(anyhow::anyhow!("Food.com dataset format has no instructions - skipping"))
    }

    fn parse_indian_food(&self, record: &csv::StringRecord, headers: &[String], idx: usize) -> Result<ValidatedRecipe> {
        let id = format!("indian_{}", idx);
        let title = self.get_field(record, headers, "RecipeName")
            .or_else(|_| self.get_field(record, headers, "TranslatedRecipeName"))?;
        
        let ingredients_raw = self.get_field(record, headers, "Ingredients")
            .or_else(|_| self.get_field(record, headers, "TranslatedIngredients"))?;
        let ingredients = self.safe_parse_list(&ingredients_raw, "ingredients")?;
        
        let instructions_raw = self.get_field(record, headers, "Instructions")
            .or_else(|_| self.get_field(record, headers, "TranslatedInstructions"))?;
        let instructions = self.safe_parse_list(&instructions_raw, "instructions")?;
        
        let cuisine = self.get_field(record, headers, "Cuisine").ok();
        let course = self.get_field(record, headers, "Course").ok();
        let total_time = self.get_field(record, headers, "TotalTimeInMins").ok();
        let cooking_time = total_time.map(|t| format!("{} minutes", t));
        let servings = self.get_field(record, headers, "Servings").ok();

        let metadata = RecipeMetadata {
            cuisine,
            course,
            cooking_time,
            servings,
            difficulty: None,
            source: Some("Indian Food Dataset".to_string()),
            original_format: "Indian Food API".to_string(),
        };

        let (mistral_input, mistral_output) = self.generate_mistral_format(&title, &ingredients, &instructions, &metadata);
        let quality_metrics = self.calculate_quality_metrics_from_parts(&title, &ingredients, &instructions);
        let quality_score = self.calculate_quality_score(&quality_metrics);

        Ok(ValidatedRecipe {
            id,
            title,
            ingredients,
            instructions,
            metadata,
            quality_score,
            mistral_input,
            mistral_output,
        })
    }

    fn parse_recipe_images(&self, record: &csv::StringRecord, headers: &[String], idx: usize) -> Result<ValidatedRecipe> {
        let id = format!("recipe_images_{}", idx);
        let title = self.get_field(record, headers, "Title")?;
        
        let ingredients_raw = self.get_field(record, headers, "Ingredients")?;
        let ingredients = self.safe_parse_list(&ingredients_raw, "ingredients")?;
        
        let instructions_raw = self.get_field(record, headers, "Instructions")?;
        let instructions = self.parse_text_instructions(&instructions_raw);
        
        let metadata = RecipeMetadata {
            cuisine: None,
            course: None,
            cooking_time: None,
            servings: None,
            difficulty: None,
            source: Some("Recipe Images Dataset".to_string()),
            original_format: "Recipe Images".to_string(),
        };

        let (mistral_input, mistral_output) = self.generate_mistral_format(&title, &ingredients, &instructions, &metadata);
        let quality_metrics = self.calculate_quality_metrics_from_parts(&title, &ingredients, &instructions);
        let quality_score = self.calculate_quality_score(&quality_metrics);

        Ok(ValidatedRecipe {
            id,
            title,
            ingredients,
            instructions,
            metadata,
            quality_score,
            mistral_input,
            mistral_output,
        })
    }

    fn parse_recipe_box(&self, record: &csv::StringRecord, headers: &[String], idx: usize) -> Result<ValidatedRecipe> {
        let id = format!("recipe_box_{}", idx);
        let title = self.get_field(record, headers, "Title")?;
        
        // Parse ingredients from multiple columns
        let mut ingredients = Vec::new();
        for i in 1..20 {
            let qty_col = if i == 1 { "Quantity".to_string() } else { format!("Quantity{:02}", i) };
            let unit_col = if i == 1 { "Unit01".to_string() } else { format!("Unit{:02}", i) };
            let ing_col = if i == 1 { "Ingredient01".to_string() } else { format!("Ingredient{:02}", i) };
            
            if let (Ok(qty), Ok(unit), Ok(ingredient)) = (
                self.get_field(record, headers, &qty_col),
                self.get_field(record, headers, &unit_col),
                self.get_field(record, headers, &ing_col),
            ) {
                if !ingredient.is_empty() && ingredient.to_lowercase() != "nan" {
                    let ingredient_text = format!("{} {} {}", qty, unit, ingredient).trim().to_string();
                    ingredients.push(ingredient_text);
                }
            }
        }
        
        let directions_raw = self.get_field(record, headers, "Directions")?;
        let instructions = self.parse_text_instructions(&directions_raw);
        
        let category = self.get_field(record, headers, "Category").ok();

        let metadata = RecipeMetadata {
            cuisine: category,
            course: None,
            cooking_time: None,
            servings: None,
            difficulty: None,
            source: Some("Recipe Box Dataset".to_string()),
            original_format: "Recipe Box".to_string(),
        };

        let (mistral_input, mistral_output) = self.generate_mistral_format(&title, &ingredients, &instructions, &metadata);
        let quality_metrics = self.calculate_quality_metrics_from_parts(&title, &ingredients, &instructions);
        let quality_score = self.calculate_quality_score(&quality_metrics);

        Ok(ValidatedRecipe {
            id,
            title,
            ingredients,
            instructions,
            metadata,
            quality_score,
            mistral_input,
            mistral_output,
        })
    }

    fn safe_parse_list(&self, text: &str, field_type: &str) -> Result<Vec<String>> {
        let trimmed = text.trim();
        
        if trimmed.is_empty() {
            return Ok(vec![]);
        }

        // Try to parse as JSON array first
        if let Ok(json_value) = serde_json::from_str::<Value>(trimmed) {
            if let Some(array) = json_value.as_array() {
                let strings: Vec<String> = array
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
                return Ok(strings);
            }
        }

        // Try to parse as Python-style list
        if trimmed.starts_with('[') && trimmed.ends_with(']') {
            let inner = &trimmed[1..trimmed.len()-1];
            let items: Vec<String> = inner
                .split(',')
                .map(|item| {
                    item.trim()
                        .trim_matches('"')
                        .trim_matches('\'')
                        .trim()
                        .to_string()
                })
                .filter(|s| !s.is_empty())
                .collect();
            return Ok(items);
        }

        // Fall back to splitting by common delimiters
        if field_type == "instructions" {
            Ok(self.parse_text_instructions(trimmed))
        } else {
            let items: Vec<String> = trimmed
                .split(&[',', '|', ';', '\n'][..])
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect();
            Ok(items)
        }
    }

    fn parse_text_instructions(&self, text: &str) -> Vec<String> {
        let trimmed = text.trim();
        
        // Split by sentence-ending periods, but be careful about abbreviations
        let sentences: Vec<String> = trimmed
            .split('.')
            .map(|s| s.trim().to_string())
            .filter(|s| s.len() > 10) // Filter out very short fragments
            .collect();
        
        if sentences.len() > 1 {
            return sentences;
        }
        
        // Split by newlines
        let lines: Vec<String> = trimmed
            .split('\n')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        
        if lines.len() > 1 {
            return lines;
        }
        
        // Return as single instruction
        vec![trimmed.to_string()]
    }

    fn generate_mistral_format(&self, title: &str, ingredients: &[String], instructions: &[String], metadata: &RecipeMetadata) -> (String, String) {
        // Mistral 7B optimized input format with clear instruction
        let mut input_parts = vec!["<s>[INST] Create a detailed recipe".to_string()];
        
        if !title.is_empty() {
            input_parts.push(format!("for {}", title));
        }
        
        if let Some(cuisine) = &metadata.cuisine {
            input_parts.push(format!("from {} cuisine", cuisine));
        }
        
        if let Some(time) = &metadata.cooking_time {
            input_parts.push(format!("that takes {}", time));
        }
        
        if let Some(servings) = &metadata.servings {
            input_parts.push(format!("serving {}", servings));
        }
        
        input_parts.push("Include ingredients list and step-by-step instructions. [/INST]".to_string());
        let input = input_parts.join(" ");
        
        // Mistral 7B output format - clean structured format
        let mut output_parts = vec![format!("# {}", title)];
        
        if !ingredients.is_empty() {
            output_parts.push("\n## Ingredients".to_string());
            for ingredient in ingredients {
                output_parts.push(format!("- {}", ingredient));
            }
        }
        
        if !instructions.is_empty() {
            output_parts.push("\n## Instructions".to_string());
            for (i, instruction) in instructions.iter().enumerate() {
                output_parts.push(format!("{}. {}", i + 1, instruction));
            }
        }
        
        // Add metadata if available
        if metadata.cuisine.is_some() || metadata.cooking_time.is_some() || metadata.servings.is_some() {
            output_parts.push("\n## Details".to_string());
            if let Some(cuisine) = &metadata.cuisine {
                output_parts.push(format!("- **Cuisine:** {}", cuisine));
            }
            if let Some(time) = &metadata.cooking_time {
                output_parts.push(format!("- **Cooking Time:** {}", time));
            }
            if let Some(servings) = &metadata.servings {
                output_parts.push(format!("- **Servings:** {}", servings));
            }
        }
        
        output_parts.push("</s>".to_string());
        let output = output_parts.join("\n");
        
        (input, output)
    }

    fn calculate_quality_metrics_from_parts(&self, title: &str, ingredients: &[String], instructions: &[String]) -> QualityMetrics {
        let has_title = !title.is_empty() && title.len() > 2;
        let has_ingredients = !ingredients.is_empty();
        let has_instructions = !instructions.is_empty();
        
        let ingredient_count = ingredients.len();
        let instruction_count = instructions.len();
        
        let total_instruction_words: usize = instructions.iter().map(|s| s.split_whitespace().count()).sum();
        let avg_instruction_length = if instruction_count > 0 {
            total_instruction_words as f64 / instruction_count as f64
        } else {
            0.0
        };
        
        let all_text = format!("{} {} {}", title, ingredients.join(" "), instructions.join(" "));
        let contains_measurements = self.measurement_regex.is_match(&all_text);
        let contains_cooking_terms = self.cooking_terms.iter().any(|term| {
            all_text.to_lowercase().contains(term)
        });
        
        let total_word_count = all_text.split_whitespace().count();
        
        // Check for junk data patterns
        let has_junk_data = self.junk_data_regex.is_match(&all_text);
        
        // Check for duplicates using content hash
        let content_hash = format!("{:x}", Sha256::digest(all_text.as_bytes()));
        let mut seen_hashes = self.seen_hashes.lock().unwrap();
        let is_duplicate = seen_hashes.contains_key(&content_hash);
        if !is_duplicate {
            seen_hashes.insert(content_hash, 1);
        }
        
        QualityMetrics {
            has_title,
            has_ingredients,
            has_instructions,
            ingredient_count,
            instruction_count,
            avg_instruction_length,
            contains_measurements,
            contains_cooking_terms,
            total_word_count,
            is_duplicate,
            has_junk_data,
        }
    }

    fn calculate_quality_metrics(&self, recipe: &ValidatedRecipe) -> QualityMetrics {
        self.calculate_quality_metrics_from_parts(&recipe.title, &recipe.ingredients, &recipe.instructions)
    }

    fn calculate_quality_score(&self, metrics: &QualityMetrics) -> f64 {
        let mut score = 0.0;
        let mut max_score = 0.0;
        
        // Essential components (60% of score)
        if metrics.has_title { score += 0.15; }
        max_score += 0.15;
        
        if metrics.has_ingredients { score += 0.25; }
        max_score += 0.25;
        
        if metrics.has_instructions { score += 0.20; }
        max_score += 0.20;
        
        // Quality indicators (30% of score)
        if metrics.ingredient_count >= 3 { score += 0.05; }
        max_score += 0.05;
        
        if metrics.instruction_count >= 2 { score += 0.05; }
        max_score += 0.05;
        
        if metrics.avg_instruction_length >= 5.0 { score += 0.05; }
        max_score += 0.05;
        
        if metrics.contains_measurements { score += 0.05; }
        max_score += 0.05;
        
        if metrics.contains_cooking_terms { score += 0.05; }
        max_score += 0.05;
        
        if metrics.total_word_count >= 50 { score += 0.05; }
        max_score += 0.05;
        
        // Penalties (20% of score)
        if metrics.is_duplicate { score -= 0.10; }
        max_score += 0.10;
        
        // Additional penalty for junk data
        if metrics.has_junk_data { score -= 0.10; }
        max_score += 0.10;
        
        if max_score > 0.0 {
            let ratio: f64 = score / max_score;
            ratio.max(0.0).min(1.0)
        } else {
            0.0
        }
    }

    fn get_field(&self, record: &csv::StringRecord, headers: &[String], field_name: &str) -> Result<String> {
        let index = headers.iter().position(|h| h.eq_ignore_ascii_case(field_name))
            .ok_or_else(|| anyhow::anyhow!("Field '{}' not found in headers: {:?}", field_name, headers))?;
        
        let value = record.get(index)
            .ok_or_else(|| anyhow::anyhow!("No value at index {} for field '{}'", index, field_name))?
            .trim()
            .to_string();
        
        if value.is_empty() {
            Err(anyhow::anyhow!("Empty value for field '{}'", field_name))
        } else {
            Ok(value)
        }
    }

    pub fn save_validated_data(&self, recipes: &[ValidatedRecipe], output_path: &str) -> Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);
        
        // Save as JSONL for efficient streaming
        for recipe in recipes {
            let json_line = serde_json::to_string(recipe)?;
            writeln!(writer, "{}", json_line)?;
        }
        
        writer.flush()?;
        println!("💾 Saved {} validated recipes to {}", recipes.len(), output_path);
        Ok(())
    }

    pub fn save_mistral_training_data(&self, recipes: &[ValidatedRecipe], output_path: &str) -> Result<()> {
        let file = File::create(output_path)?;
        let mut writer = BufWriter::new(file);
        
        for recipe in recipes {
            let training_example = json!({
                "input": recipe.mistral_input,
                "output": recipe.mistral_output,
                "quality_score": recipe.quality_score,
                "metadata": recipe.metadata
            });
            writeln!(writer, "{}", serde_json::to_string(&training_example)?)?;
        }
        
        writer.flush()?;
        println!("🎯 Saved {} Mistral 7B training examples to {}", recipes.len(), output_path);
        Ok(())
    }

    pub fn print_validation_report(&self, stats: &ValidationStats) {
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_header(vec!["Metric", "Count", "Percentage"]);

        let total = stats.total_processed as f64;
        
        table.add_row(vec![
            "📊 Total Processed".to_string(),
            format!("{}", stats.total_processed),
            "100.0%".to_string(),
        ]);
        
        table.add_row(vec![
            "✅ Valid Recipes".to_string(),
            format!("{}", stats.valid_recipes),
            format!("{:.1}%", (stats.valid_recipes as f64 / total) * 100.0),
        ]);
        
        table.add_row(vec![
            "🔄 Duplicates Found".to_string(),
            format!("{}", stats.duplicates_found),
            format!("{:.1}%", (stats.duplicates_found as f64 / total) * 100.0),
        ]);
        
        table.add_row(vec![
            "❌ Parsing Errors".to_string(),
            format!("{}", stats.parsing_errors),
            format!("{:.1}%", (stats.parsing_errors as f64 / total) * 100.0),
        ]);
        
        table.add_row(vec![
            "⚠️ Quality Failures".to_string(),
            format!("{}", stats.quality_failures),
            format!("{:.1}%", (stats.quality_failures as f64 / total) * 100.0),
        ]);
        
        table.add_row(vec![
            "🗑️ Junk Data Found".to_string(),
            format!("{}", stats.junk_data_found),
            format!("{:.1}%", (stats.junk_data_found as f64 / total) * 100.0),
        ]);

        println!("\n📋 VALIDATION REPORT");
        println!("{}", table);

        if !stats.quality_distribution.is_empty() {
            println!("\n🎯 QUALITY DISTRIBUTION");
            let mut quality_table = Table::new();
            quality_table
                .load_preset(UTF8_FULL)
                .apply_modifier(UTF8_ROUND_CORNERS)
                .set_header(vec!["Quality Tier", "Count", "Percentage"]);

            for (tier, count) in &stats.quality_distribution {
                quality_table.add_row(vec![
                    tier.clone(),
                    format!("{}", count),
                    format!("{:.1}%", (*count as f64 / stats.valid_recipes as f64) * 100.0),
                ]);
            }
            println!("{}", quality_table);
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let matches = Command::new("Recipe Data Validator")
        .version("2.0")
        .about("Enterprise-grade recipe data validation and cleaning pipeline")
        .arg(
            Arg::new("input")
                .short('i')
                .long("input")
                .value_name("FILE")
                .help("Input file path (CSV or JSONL)")
                .required(true),
        )
        .arg(
            Arg::new("mode")
                .short('m')
                .long("mode")
                .value_name("MODE")
                .help("Processing mode")
                .required(true)
                .value_parser(["csv", "jsonl"]),
        )
        .arg(
            Arg::new("format")
                .short('f')
                .long("format")
                .value_name("FORMAT")
                .help("Dataset format (only for CSV mode)")
                .value_parser(["recipe_nlg", "food_com_raw", "food_com_dataset", "indian_food", "recipe_images", "recipe_box"]),
        )
        .arg(
            Arg::new("output")
                .short('o')
                .long("output")
                .value_name("FILE")
                .help("Output file for FLAN-T5 training data")
                .default_value("flan_t5_output.jsonl"),
        )
        .arg(
            Arg::new("min-quality")
                .short('q')
                .long("min-quality")
                .value_name("SCORE")
                .help("Minimum quality score (0.0-1.0)")
                .default_value("0.6"),
        )
        .get_matches();

    let input_file = matches.get_one::<String>("input").unwrap();
    let mode = matches.get_one::<String>("mode").unwrap();
    let output_file = matches.get_one::<String>("output").unwrap();
    let min_quality: f64 = matches.get_one::<String>("min-quality")
        .unwrap()
        .parse()
        .unwrap_or(0.6);

    match mode.as_str() {
        "jsonl" => {
            // Convert cleaned JSONL to FLAN-T5 format
            jsonl_converter::convert_cleaned_to_flan_t5(input_file, output_file, min_quality)?;
        }
        "csv" => {
            // Original CSV validation pipeline
            let format = matches.get_one::<String>("format")
                .ok_or_else(|| anyhow::anyhow!("--format required for CSV mode"))?;

            println!("🚀 Starting Recipe Data Validation Pipeline");
            println!("📂 Input: {}", input_file);
            println!("📋 Format: {}", format);
            println!("💾 Output: {}", output_file);

            let validator = RecipeValidator::new();
            let (validated_recipes, stats) = validator.validate_csv(input_file, format)?;
            validator.print_validation_report(&stats);
            validator.save_mistral_training_data(&validated_recipes, output_file)?;

            println!("\n🎉 Validation pipeline completed successfully!");
            println!("✨ {} high-quality recipes ready for training", validated_recipes.len());
        }
        _ => {
            return Err(anyhow::anyhow!("Invalid mode: {}", mode));
        }
    }

    Ok(())
}