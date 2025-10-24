use anyhow::{Context, Result};
use ahash::AHashSet;
use clap::Parser;
use csv::{Reader, Writer};
use indicatif::{ProgressBar, ProgressStyle};
use lazy_static::lazy_static;
use rayon::prelude::*;
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

#[derive(Parser, Debug)]
#[command(author, version, about = "High-performance recipe dataset cleaner", long_about = None)]
struct Args {
    /// Input file or directory
    #[arg(short, long)]
    input: PathBuf,

    /// Output file or directory
    #[arg(short, long)]
    output: Option<PathBuf>,

    /// Process directories recursively
    #[arg(short, long)]
    recursive: bool,

    /// Dry run - don't write files
    #[arg(long)]
    dry_run: bool,

    /// Verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Minimum ingredients count
    #[arg(long, default_value = "1")]
    min_ingredients: usize,

    /// Minimum instructions length (total after joining all steps)
    #[arg(long, default_value = "10")]
    min_instructions_len: usize,

    /// Maximum special character ratio
    #[arg(long, default_value = "0.3")]
    max_special_char_ratio: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Recipe {
    #[serde(flatten)]
    data: serde_json::Map<String, Value>,
}

#[derive(Debug, Default)]
struct CleaningStats {
    total_processed: AtomicUsize,
    kept_clean: AtomicUsize,
    removed_garbage: AtomicUsize,
    removed_too_short: AtomicUsize,
    removed_missing_fields: AtomicUsize,
    removed_duplicates: AtomicUsize,
    fixed_formatting: AtomicUsize,
}

impl CleaningStats {
    fn print(&self) {
        let total = self.total_processed.load(Ordering::Relaxed);
        let kept = self.kept_clean.load(Ordering::Relaxed);
        let garbage = self.removed_garbage.load(Ordering::Relaxed);
        let short = self.removed_too_short.load(Ordering::Relaxed);
        let missing = self.removed_missing_fields.load(Ordering::Relaxed);
        let duplicates = self.removed_duplicates.load(Ordering::Relaxed);
        let fixed = self.fixed_formatting.load(Ordering::Relaxed);

        println!("\n{}", "=".repeat(60));
        println!("📊 CLEANING STATISTICS");
        println!("{}", "=".repeat(60));
        println!("Total recipes processed:     {:>10}", format_number(total));
        println!("Kept (clean):               {:>10} ✅", format_number(kept));
        println!("Fixed formatting issues:    {:>10} 🔧", format_number(fixed));
        println!("Removed (garbage):          {:>10} 🗑️", format_number(garbage));
        println!("Removed (too short):        {:>10} 📏", format_number(short));
        println!("Removed (missing fields):   {:>10} ❓", format_number(missing));
        println!("Removed (duplicates):       {:>10} 🔁", format_number(duplicates));
        println!();

        if total > 0 {
            let kept_pct = (kept as f64 / total as f64) * 100.0;
            println!("Quality rate: {:.1}% kept, {:.1}% removed", kept_pct, 100.0 - kept_pct);
        }
    }
}

fn format_number(n: usize) -> String {
    n.to_string()
        .as_bytes()
        .rchunks(3)
        .rev()
        .map(|chunk| std::str::from_utf8(chunk).unwrap())
        .collect::<Vec<_>>()
        .join(",")
}

lazy_static! {
    // Garbage patterns
    static ref GARBAGE_PATTERNS: Vec<Regex> = vec![
        Regex::new(r":\)+").unwrap(),
        Regex::new(r":\(+").unwrap(),
        Regex::new(r"\blol\b").unwrap(),
        Regex::new(r"\bhaha+\b").unwrap(),
        Regex::new(r"\bwtf\b").unwrap(),
        Regex::new(r"\bomg\b").unwrap(),
        Regex::new(r"\btest\b.*recipe").unwrap(),
        Regex::new(r"\bgarbage\b").unwrap(),
        Regex::new(r"\bxxx+\b").unwrap(),
        Regex::new(r"\basdf+\b").unwrap(),
        Regex::new(r"\b(fuck|shit|damn)\b").unwrap(),
        Regex::new(r"[!?]{3,}").unwrap(),
        Regex::new(r"\.{4,}").unwrap(),
        // Can't use backreferences in Rust regex, check manually instead
        Regex::new(r"\b\d{10,}\b").unwrap(),
    ];

    // Excel date fixes
    static ref EXCEL_DATE_FIXES: Vec<(Regex, &'static str)> = vec![
        (Regex::new(r"\b(\d+)-Jan\b").unwrap(), "1/2"),
        (Regex::new(r"\b2-Feb\b").unwrap(), "2/3"),
        (Regex::new(r"\b3-Feb\b").unwrap(), "3/4"),
        (Regex::new(r"\b2-Mar\b").unwrap(), "1/3"),
        (Regex::new(r"\b3-Mar\b").unwrap(), "3/4"),
    ];
}

struct RecipeCleaner {
    args: Args,
    stats: CleaningStats,
    seen_fingerprints: Mutex<AHashSet<String>>,
}

impl RecipeCleaner {
    fn new(args: Args) -> Self {
        Self {
            args,
            stats: CleaningStats::default(),
            seen_fingerprints: Mutex::new(AHashSet::new()),
        }
    }

    fn is_garbage(&self, text: &str) -> bool {
        if text.is_empty() {
            return true;
        }

        let text_lower = text.to_lowercase();

        // Check garbage patterns
        for pattern in GARBAGE_PATTERNS.iter() {
            if pattern.is_match(&text_lower) {
                return true;
            }
        }

        // Check for repeated characters (aaaaaaa, 1111111, etc.)
        let chars: Vec<char> = text.chars().collect();
        let mut repeated_count = 1;
        for window in chars.windows(2) {
            if window[0] == window[1] && window[0].is_alphanumeric() {
                repeated_count += 1;
                if repeated_count >= 6 {
                    return true;
                }
            } else {
                repeated_count = 1;
            }
        }

        // Check special character ratio
        let special_chars = text.chars().filter(|c| {
            !c.is_alphanumeric() && !c.is_whitespace() &&
            *c != '.' && *c != ',' && *c != '!' && *c != '?' &&
            *c != ';' && *c != ':' && *c != '-' && *c != '\'' &&
            *c != '"' && *c != '(' && *c != ')' && *c != '[' && *c != ']'
        }).count();

        let special_ratio = special_chars as f32 / text.len() as f32;
        if special_ratio > self.args.max_special_char_ratio {
            return true;
        }

        // Check minimum words
        let words: Vec<_> = text.split_whitespace()
            .filter(|w| w.chars().filter(|c| c.is_alphabetic()).count() >= 2)
            .collect();

        if words.len() < 3 {
            return true;
        }

        // Check excessive capitalization
        if text.len() > 10 {
            let caps_count = text.chars().filter(|c| c.is_uppercase()).count();
            let caps_ratio = caps_count as f32 / text.len() as f32;
            if caps_ratio > 0.8 {
                return true;
            }
        }

        false
    }

    fn fix_excel_dates(&self, text: &str) -> String {
        let mut result = text.to_string();

        // Simple replacements for common Excel date conversions
        result = result.replace("2-Jan", "1/2");
        result = result.replace("3-Jan", "1/3");
        result = result.replace("4-Jan", "1/4");
        result = result.replace("2-Feb", "2/3");
        result = result.replace("3-Feb", "3/4");
        result = result.replace("2-Mar", "1/3");

        result
    }

    fn clean_text(&self, text: &str) -> String {
        let mut cleaned = self.fix_excel_dates(text);

        // Fix encoding issues
        cleaned = cleaned.replace('\u{00b0}', "°");
        cleaned = cleaned.replace('\u{00bd}', "1/2");
        cleaned = cleaned.replace('\u{00bc}', "1/4");
        cleaned = cleaned.replace('\u{00be}', "3/4");

        // Remove zero-width characters
        cleaned = cleaned.chars()
            .filter(|c| !matches!(c, '\u{200b}' | '\u{200c}' | '\u{200d}' | '\u{feff}'))
            .collect();

        // Normalize whitespace
        cleaned.split_whitespace().collect::<Vec<_>>().join(" ")
    }

    fn get_field_value<'a>(&self, recipe: &'a serde_json::Map<String, Value>, field_names: &[&str]) -> Option<&'a Value> {
        for name in field_names {
            if let Some(val) = recipe.get(*name) {
                return Some(val);
            }
        }
        None
    }

    fn value_to_string_list(&self, value: &Value) -> Vec<String> {
        match value {
            Value::String(s) => {
                // Try parsing as JSON array first
                if let Ok(Value::Array(arr)) = serde_json::from_str::<Value>(s) {
                    arr.iter()
                        .filter_map(|v| v.as_str())
                        .map(|s| self.clean_text(s))
                        .filter(|s| !s.is_empty())
                        .collect()
                } else {
                    // Split by common delimiters (including pipe for datasets like food_recipes)
                    s.split(&[',', ';', '\n', '|'][..])
                        .map(|s| self.clean_text(s.trim()))
                        .filter(|s| !s.is_empty())
                        .collect()
                }
            }
            Value::Array(arr) => {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .map(|s| self.clean_text(s))
                    .filter(|s| !s.is_empty())
                    .collect()
            }
            _ => vec![],
        }
    }

    fn get_recipe_fingerprint(&self, recipe: &serde_json::Map<String, Value>) -> String {
        let title_fields = ["title", "name", "recipe_name", "recipe_title", "Title", "RecipeName", "RecipeTitle"];
        let ing_fields = ["ingredients", "Ingredients", "ingredient_tokens", "RecipeIngredientParts"];

        let title = self.get_field_value(recipe, &title_fields)
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_lowercase();

        let ingredients = self.get_field_value(recipe, &ing_fields)
            .map(|v| {
                self.value_to_string_list(v)
                    .iter()
                    .take(3)
                    .map(|s| s.to_lowercase())
                    .collect::<Vec<_>>()
                    .join(" ")
            })
            .unwrap_or_default();

        format!("{}::{}", title, ingredients)
    }

    fn validate_and_clean_recipe(&self, recipe: &mut serde_json::Map<String, Value>) -> Result<bool, String> {
        self.stats.total_processed.fetch_add(1, Ordering::Relaxed);

        // Handle nested recipe_nlg format: flatten input_data and output_data
        if recipe.contains_key("input_data") && recipe.contains_key("output_data") {
            let mut flattened = serde_json::Map::new();

            // Extract ingredients from input_data
            if let Some(Value::Object(input_data)) = recipe.get("input_data") {
                if let Some(ingredients) = input_data.get("ingredients") {
                    flattened.insert("ingredients".to_string(), ingredients.clone());
                }
            }

            // Extract title and instructions from output_data
            if let Some(Value::Object(output_data)) = recipe.get("output_data") {
                // In recipe_nlg, the title is a number and instructions contains the actual title
                // We need to swap these
                if let Some(instructions) = output_data.get("instructions") {
                    if let Value::Array(inst_array) = instructions {
                        if !inst_array.is_empty() {
                            if let Some(actual_title) = inst_array[0].as_str() {
                                flattened.insert("title".to_string(), Value::String(actual_title.to_string()));
                            }
                        }
                    }
                }
                // Also keep other output_data fields
                for (k, v) in output_data.iter() {
                    if k != "instructions" && k != "title" {
                        flattened.insert(k.clone(), v.clone());
                    }
                }
            }

            // Replace recipe with flattened version
            *recipe = flattened;
        }

        // Field name variants (expanded for all dataset types)
        let title_fields = ["title", "name", "recipe_name", "recipe_title", "Title", "RecipeName", "RecipeTitle"];
        let ing_fields = ["ingredients", "Ingredients", "ingredient_tokens", "RecipeIngredientParts", "RecipeIngredientQuantities"];
        let inst_fields = ["instructions", "directions", "steps", "Instructions", "RecipeInstructions"];

        // Only reject if completely missing title (truly broken record)
        let has_title = self.get_field_value(recipe, &title_fields).is_some();
        if !has_title {
            self.stats.removed_missing_fields.fetch_add(1, Ordering::Relaxed);
            return Err("missing title".to_string());
        }

        // Check for duplicates
        let fingerprint = self.get_recipe_fingerprint(recipe);
        {
            let mut seen = self.seen_fingerprints.lock().unwrap();
            if seen.contains(&fingerprint) {
                self.stats.removed_duplicates.fetch_add(1, Ordering::Relaxed);
                return Err("duplicate".to_string());
            }
            seen.insert(fingerprint);
        }

        // Clean ALL string fields - remove garbage tokens but KEEP the recipe
        for (_key, value) in recipe.iter_mut() {
            if let Some(s) = value.as_str() {
                let mut cleaned = self.clean_text(s);

                // Remove garbage tokens (emoticons, slang, etc.)
                cleaned = cleaned.replace(":)))", "");
                cleaned = cleaned.replace(":))", "");
                cleaned = cleaned.replace(":(((", "");
                cleaned = cleaned.replace(":((", "");
                cleaned = cleaned.replace(" lol ", " ");
                cleaned = cleaned.replace(" lmao ", " ");
                cleaned = cleaned.replace(" wtf ", " ");
                cleaned = cleaned.replace(" omg ", " ");
                cleaned = cleaned.replace("!!!", "!");
                cleaned = cleaned.replace("???", "?");
                cleaned = cleaned.replace("....", "...");

                // Clean up multiple spaces
                cleaned = cleaned.split_whitespace().collect::<Vec<_>>().join(" ");

                if cleaned != s {
                    self.stats.fixed_formatting.fetch_add(1, Ordering::Relaxed);
                    *value = Value::String(cleaned);
                }
            }
        }

        self.stats.kept_clean.fetch_add(1, Ordering::Relaxed);
        Ok(true)
    }

    fn clean_json_file(&self, input_path: &Path, output_path: &Path) -> Result<usize> {
        println!("\n🧹 Cleaning: {}", input_path.display());

        let file = File::open(input_path)?;
        let file_size = file.metadata()?.len();

        let pb = ProgressBar::new(file_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} {msg}")
                .unwrap()
        );

        let mut cleaned_recipes = Vec::new();
        let mut bytes_read = 0usize;

        // Try JSONL format (one JSON object per line)
        let reader = BufReader::new(file);
        let mut is_jsonl = false;

        for line in reader.lines() {
            let line = line?;
            bytes_read += line.len() + 1;
            pb.set_position(bytes_read as u64);

            if line.trim().is_empty() {
                continue;
            }

            // Parse line as JSON
            if let Ok(Value::Object(mut recipe)) = serde_json::from_str(&line) {
                is_jsonl = true;
                if self.validate_and_clean_recipe(&mut recipe).is_ok() {
                    cleaned_recipes.push(Value::Object(recipe));
                }
            }
        }

        pb.finish_with_message("Done");

        // If no recipes found via JSONL, try streaming JSON array
        if cleaned_recipes.is_empty() && !is_jsonl {
            println!("  Trying as JSON array with streaming...");

            let file = File::open(input_path)?;
            let reader = BufReader::new(file);

            // Use streaming deserializer for large JSON arrays
            let stream = serde_json::Deserializer::from_reader(reader).into_iter::<Value>();

            let pb2 = ProgressBar::new_spinner();
            pb2.set_style(
                ProgressStyle::default_spinner()
                    .template("{spinner:.green} [{elapsed_precise}] {msg}")
                    .unwrap()
            );

            let mut count = 0;
            for value in stream {
                if let Ok(Value::Array(recipes)) = value {
                    // Process array elements
                    for recipe_val in recipes {
                        count += 1;
                        if count % 1000 == 0 {
                            pb2.set_message(format!("Processed {} recipes", format_number(count)));
                        }

                        if let Value::Object(mut recipe) = recipe_val {
                            if self.validate_and_clean_recipe(&mut recipe).is_ok() {
                                cleaned_recipes.push(Value::Object(recipe));
                            }
                        }
                    }
                }
            }

            pb2.finish_with_message(format!("Processed {} recipes", format_number(count)));
        }

        // Write output
        if !self.args.dry_run && !cleaned_recipes.is_empty() {
            let output_file = File::create(output_path)?;
            let mut writer = BufWriter::new(output_file);

            // Write as JSONL for efficiency
            for recipe in &cleaned_recipes {
                serde_json::to_writer(&mut writer, recipe)?;
                writeln!(writer)?;
            }
        }

        let count = cleaned_recipes.len();
        println!("  ✅ Kept {} clean recipes", format_number(count));

        Ok(count)
    }

    fn clean_csv_file(&self, input_path: &Path, output_path: &Path) -> Result<usize> {
        println!("\n🧹 Cleaning: {}", input_path.display());

        let file = File::open(input_path)?;
        let file_size = file.metadata()?.len();

        let pb = ProgressBar::new(file_size);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {bar:40.cyan/blue} {bytes}/{total_bytes} {msg}")
                .unwrap()
        );

        let mut rdr = Reader::from_reader(file);
        let headers = rdr.headers()?.clone();

        let records: Vec<_> = rdr.records()
            .filter_map(|r| r.ok())
            .collect();

        // Process in parallel
        let cleaned_records: Vec<_> = records
            .par_iter()
            .filter_map(|record| {
                // Convert CSV record to JSON object
                let mut recipe_map = serde_json::Map::new();
                for (i, header) in headers.iter().enumerate() {
                    if let Some(value) = record.get(i) {
                        recipe_map.insert(header.to_string(), Value::String(value.to_string()));
                    }
                }

                if self.validate_and_clean_recipe(&mut recipe_map).is_ok() {
                    // Convert back to CSV record
                    let mut new_record = csv::StringRecord::new();
                    for header in headers.iter() {
                        let value = recipe_map.get(header)
                            .and_then(|v| v.as_str())
                            .unwrap_or("");
                        new_record.push_field(value);
                    }
                    Some(new_record)
                } else {
                    None
                }
            })
            .collect();

        pb.finish_with_message("Done");

        // Write output
        if !self.args.dry_run && !cleaned_records.is_empty() {
            let output_file = File::create(output_path)?;
            let mut wtr = Writer::from_writer(output_file);

            wtr.write_record(&headers)?;
            for record in &cleaned_records {
                wtr.write_record(record)?;
            }
            wtr.flush()?;
        }

        let count = cleaned_records.len();
        println!("  ✅ Kept {} clean recipes", format_number(count));

        Ok(count)
    }

    fn clean_file(&self, input_path: &Path) -> Result<usize> {
        // Clear fingerprints for each file to avoid cross-file duplicate detection
        {
            let mut seen = self.seen_fingerprints.lock().unwrap();
            seen.clear();
        }

        let output_path = if let Some(ref out) = self.args.output {
            // Check if output is meant to be a directory (exists and is dir, OR ends with /)
            let is_output_dir = out.is_dir() || out.to_str().map(|s| s.ends_with('/')).unwrap_or(false);
            if is_output_dir || self.args.recursive {
                // Canonicalize paths to handle relative paths correctly
                let input_abs = input_path.canonicalize().unwrap_or_else(|_| input_path.to_path_buf());
                let input_base_abs = self.args.input.canonicalize().unwrap_or_else(|_| self.args.input.clone());

                // Preserve subdirectory structure
                let relative_path = if let Ok(rel) = input_abs.strip_prefix(&input_base_abs) {
                    rel
                } else {
                    // Fallback: just use the filename
                    input_path.file_name().map(|n| Path::new(n)).unwrap()
                };

                let filename = relative_path.file_stem().unwrap().to_str().unwrap();
                let ext = relative_path.extension().unwrap().to_str().unwrap();

                // Preserve directory structure
                if let Some(parent) = relative_path.parent() {
                    if parent.as_os_str().is_empty() {
                        out.join(format!("{}_cleaned.{}", filename, ext))
                    } else {
                        out.join(parent).join(format!("{}_cleaned.{}", filename, ext))
                    }
                } else {
                    out.join(format!("{}_cleaned.{}", filename, ext))
                }
            } else {
                out.clone()
            }
        } else {
            let filename = input_path.file_stem().unwrap().to_str().unwrap();
            let ext = input_path.extension().unwrap().to_str().unwrap();
            input_path.with_file_name(format!("{}_cleaned.{}", filename, ext))
        };

        // Create output directory if needed
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        match input_path.extension().and_then(|s| s.to_str()) {
            Some("json") | Some("jsonl") => self.clean_json_file(input_path, &output_path),
            Some("csv") => self.clean_csv_file(input_path, &output_path),
            _ => {
                println!("  ⚠️  Unsupported file type");
                Ok(0)
            }
        }
    }

    fn find_dataset_files(&self, dir: &Path) -> Vec<PathBuf> {
        let mut files = Vec::new();

        if self.args.verbose {
            println!("🔍 Scanning directory: {}", dir.display());
        }

        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();

                if path.is_file() {
                    if let Some(ext) = path.extension() {
                        if ext == "json" || ext == "jsonl" || ext == "csv" {
                            // Skip tiny files
                            if let Ok(metadata) = path.metadata() {
                                if metadata.len() > 1024 {
                                    if self.args.verbose {
                                        println!("  ✓ Found file: {}", path.display());
                                    }
                                    files.push(path);
                                }
                            }
                        }
                    }
                } else if path.is_dir() && self.args.recursive {
                    if self.args.verbose {
                        println!("  📁 Recursing into: {}", path.display());
                    }
                    files.extend(self.find_dataset_files(&path));
                }
            }
        }

        files
    }

    fn run(&self) -> Result<()> {
        if self.args.dry_run {
            println!("🔍 DRY RUN MODE - No files will be modified\n");
        }

        if self.args.input.is_file() {
            self.clean_file(&self.args.input)?;
        } else if self.args.input.is_dir() {
            let files = self.find_dataset_files(&self.args.input);
            println!("📁 Found {} dataset files to clean\n", files.len());

            for file in files {
                if let Err(e) = self.clean_file(&file) {
                    eprintln!("  ❌ Error cleaning {}: {}", file.display(), e);
                }
            }
        } else {
            anyhow::bail!("Invalid input path");
        }

        self.stats.print();

        Ok(())
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let cleaner = RecipeCleaner::new(args);
    cleaner.run()
}
