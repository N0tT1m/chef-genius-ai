// Simplified validation - just clean garbage tokens, keep almost all recipes

fn validate_and_clean_recipe(&self, recipe: &mut serde_json::Map<String, Value>) -> Result<bool, String> {
    self.stats.total_processed.fetch_add(1, Ordering::Relaxed);

    // Field name variants
    let title_fields = ["title", "name", "recipe_name", "recipe_title", "Title", "RecipeName", "RecipeTitle"];

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
