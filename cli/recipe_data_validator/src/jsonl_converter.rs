use anyhow::Result;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use indicatif::{ProgressBar, ProgressStyle};
use std::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug, Deserialize)]
struct CleanedRecipe {
    #[serde(default)]
    title: String,
    #[serde(default)]
    ingredients: Value,
    #[serde(default)]
    instructions: Value,
}

#[derive(Debug, Serialize)]
struct FlanT5Recipe {
    input: String,
    output: String,
    quality_score: f64,
    metadata: serde_json::Map<String, Value>,
}

pub fn convert_cleaned_to_flan_t5(
    input_path: &str,
    output_path: &str,
    min_quality: f64,
) -> Result<usize> {
    println!("🚀 Converting cleaned JSONL to FLAN-T5 format");
    println!("📂 Input: {}", input_path);
    println!("💾 Output: {}", output_path);
    println!("🎯 Min quality: {}", min_quality);

    // Count lines first
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let total_lines = reader.lines().count();
    println!("📊 Total recipes: {}", total_lines);

    // Read all lines into memory for parallel processing
    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let lines: Vec<String> = reader.lines().filter_map(|l| l.ok()).collect();

    let pb = ProgressBar::new(total_lines as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({per_sec}, {eta})")
            .unwrap()
            .progress_chars("#>-"),
    );

    let kept = AtomicUsize::new(0);
    let filtered = AtomicUsize::new(0);

    // Process in parallel
    let flan_t5_recipes: Vec<FlanT5Recipe> = lines
        .par_iter()
        .filter_map(|line| {
            pb.inc(1);

            if line.trim().is_empty() {
                return None;
            }

            // Parse cleaned recipe
            let cleaned: CleanedRecipe = match serde_json::from_str(line) {
                Ok(r) => r,
                Err(_) => return None,
            };

            // Convert to FLAN-T5 format
            match convert_recipe(&cleaned) {
                Some(recipe) => {
                    if recipe.quality_score >= min_quality {
                        kept.fetch_add(1, Ordering::Relaxed);
                        Some(recipe)
                    } else {
                        filtered.fetch_add(1, Ordering::Relaxed);
                        None
                    }
                }
                None => {
                    filtered.fetch_add(1, Ordering::Relaxed);
                    None
                }
            }
        })
        .collect();

    pb.finish_with_message("✅ Conversion complete");

    // Write output
    let output_file = File::create(output_path)?;
    let mut writer = BufWriter::new(output_file);

    for recipe in &flan_t5_recipes {
        serde_json::to_writer(&mut writer, recipe)?;
        writeln!(writer)?;
    }

    writer.flush()?;

    let kept_count = kept.load(Ordering::Relaxed);
    let filtered_count = filtered.load(Ordering::Relaxed);
    let total = kept_count + filtered_count;

    println!("\n✅ Conversion complete!");
    println!("   Processed: {}", total);
    println!("   Kept: {} ({:.1}%)", kept_count, (kept_count as f64 / total as f64) * 100.0);
    println!("   Filtered: {} ({:.1}%)", filtered_count, (filtered_count as f64 / total as f64) * 100.0);

    Ok(kept_count)
}

fn convert_recipe(recipe: &CleanedRecipe) -> Option<FlanT5Recipe> {
    // Extract title
    let title = recipe.title.trim();
    if title.is_empty() {
        return None;
    }

    // Parse ingredients
    let ingredients = parse_list(&recipe.ingredients);
    if ingredients.is_empty() {
        return None;
    }

    // Parse instructions (can be empty for some datasets like recipe_nlg)
    let instructions = parse_list(&recipe.instructions);

    // Create FLAN-T5 format - make instructions mandatory by generating them from ingredients if needed
    let input = format!("Create a detailed recipe for {} with step-by-step instructions.", title);

    let mut output_parts = vec![format!("# {}", title), String::new()];

    // Add ingredients
    output_parts.push("## Ingredients".to_string());
    for ing in &ingredients {
        if !ing.is_empty() {
            output_parts.push(format!("- {}", ing));
        }
    }

    // Add instructions - if missing, generate basic ones from title
    let final_instructions = if instructions.is_empty() {
        vec![
            format!("Gather all ingredients for {}.", title),
            "Combine ingredients according to your preferred method.".to_string(),
            "Cook or prepare as appropriate.".to_string(),
            "Serve and enjoy!".to_string(),
        ]
    } else {
        instructions
    };

    output_parts.push(String::new());
    output_parts.push("## Instructions".to_string());
    for (i, inst) in final_instructions.iter().enumerate() {
        if !inst.is_empty() {
            output_parts.push(format!("{}. {}", i + 1, inst));
        }
    }

    let output = output_parts.join("\n");

    // Calculate quality score (before instructions are moved)
    let quality_score = calculate_quality_score(&ingredients, &final_instructions);

    // Create metadata
    let mut metadata = serde_json::Map::new();
    metadata.insert("title".to_string(), json!(title));
    metadata.insert("ingredient_count".to_string(), json!(ingredients.len()));
    metadata.insert("instruction_count".to_string(), json!(final_instructions.len()));

    Some(FlanT5Recipe {
        input,
        output,
        quality_score,
        metadata,
    })
}

fn parse_list(value: &Value) -> Vec<String> {
    match value {
        Value::Array(arr) => arr
            .iter()
            .filter_map(|v| v.as_str())
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect(),
        Value::String(s) => {
            // Try parsing as JSON array
            if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(s) {
                return arr
                    .iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }

            // Split by common delimiters
            s.split(&[',', '|', '\n'][..])
                .map(|item| item.trim().to_string())
                .filter(|item| !item.is_empty())
                .collect()
        }
        _ => vec![],
    }
}

fn calculate_quality_score(ingredients: &[String], instructions: &[String]) -> f64 {
    let mut score: f64 = 0.0;

    // Has ingredients (most important)
    if !ingredients.is_empty() {
        score += 0.5;
        // Bonus for good count
        if ingredients.len() >= 3 {
            score += 0.2;
        }
        if ingredients.len() >= 5 {
            score += 0.1;
        }
    }

    // Has instructions (less important since we generate them if missing)
    if !instructions.is_empty() {
        score += 0.1;
        // Bonus for detailed instructions
        if instructions.len() >= 3 {
            score += 0.1;
        }
    }

    if score > 1.0 { 1.0 } else { score }
}
