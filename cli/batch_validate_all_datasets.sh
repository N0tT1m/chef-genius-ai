#!/bin/bash

# Batch validation script for all recipe datasets
echo "🚀 Starting batch validation of all recipe datasets"

# Detect execution context and set paths accordingly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "$SCRIPT_DIR" == */cli ]]; then
    # Running from cli directory
    echo "🏠 Running from cli directory"
    VALIDATOR="./recipe_data_validator/target/release/validate_recipes"
    OUTPUT_DIR="validated_datasets"
    DATA_PREFIX="../data"
else
    # Running from root directory (Docker context)
    echo "🐳 Running from root directory (Docker context)"
    VALIDATOR="./cli/recipe_data_validator/target/release/validate_recipes"
    OUTPUT_DIR="cli/validated_datasets"
    DATA_PREFIX="data"
fi

mkdir -p "$OUTPUT_DIR"

# Function to validate a dataset
validate_dataset() {
    local input_file="$1"
    local format="$2"
    local dataset_name="$3"
    
    echo "📊 Processing: $dataset_name"
    echo "   Input: $input_file"
    echo "   Format: $format"
    
    local output_file="$OUTPUT_DIR/${dataset_name}_validated.jsonl"
    local training_file="$OUTPUT_DIR/${dataset_name}_flan_t5.jsonl"
    
    if [ -f "$input_file" ]; then
        $VALIDATOR -i "$input_file" -f "$format" -o "$output_file" -t "$training_file"
        echo "✅ Completed: $dataset_name"
        echo "   Output: $output_file"
        echo "   Training: $training_file"
        echo ""
    else
        echo "❌ File not found: $input_file"
        echo ""
    fi
}

# Validate all major datasets
echo "🎯 Validating RecipeNLG (2.2M recipes)..."
validate_dataset "$DATA_PREFIX/datasets/recipe_nlg/RecipeNLG_dataset.csv" "recipe_nlg" "recipe_nlg"

echo "🎯 Validating Food.com RAW (267K recipes)..."
validate_dataset "$DATA_PREFIX/datasets/food_com_recipes_2m/RAW_recipes.csv" "food_com_raw" "food_com_raw"

echo "🎯 Validating Indian Food Dataset (6K recipes)..."
validate_dataset "$DATA_PREFIX/datasets/indian_food_6k/IndianFoodDatasetCSV.csv" "indian_food" "indian_food_6k"

echo "🎯 Validating Recipe Images Dataset..."
validate_dataset "$DATA_PREFIX/datasets/recipe_ingredient_images/Food Ingredients and Recipe Dataset with Image Name Mapping.csv" "recipe_images" "recipe_images"

echo "🎯 Validating Recipe Box Dataset..."
validate_dataset "$DATA_PREFIX/datasets/recipe_box/dataset.csv" "recipe_box" "recipe_box"

echo "🎯 Validating 13K Recipes Dataset..."
validate_dataset "$DATA_PREFIX/datasets/recipe_dataset_simple/13k-recipes.csv" "recipe_dataset_simple" "13k_recipes"

echo "🎯 Validating Food Recipes 8K..."
validate_dataset "$DATA_PREFIX/datasets/food_recipes_8k/food_recipes.csv" "recipe_dataset_simple" "food_recipes_8k"

echo "🎯 Validating Epicurious Recipes..."
validate_dataset "$DATA_PREFIX/datasets/epi_r.csv" "recipe_dataset_simple" "epicurious"

echo "🎯 Validating PP Recipes..."
validate_dataset "$DATA_PREFIX/datasets/PP_recipes.csv" "recipe_dataset_simple" "pp_recipes"

# Additional Indian food datasets
echo "🎯 Validating Indian Recipe API..."
validate_dataset "$DATA_PREFIX/datasets/indian_recipe_api/dataset.csv" "indian_food" "indian_recipe_api"

echo "🎯 Validating Indian Food Analysis..."
validate_dataset "$DATA_PREFIX/datasets/indian_food_analysis/dataset.csv" "indian_food" "indian_food_analysis"

# Combine all training files
echo "🔄 Combining all validated datasets..."
cat "$OUTPUT_DIR"/*_flan_t5.jsonl > "$OUTPUT_DIR/combined_all_datasets_flan_t5.jsonl"

# Count totals
echo "📊 FINAL VALIDATION SUMMARY"
echo "================================="
for file in "$OUTPUT_DIR"/*_flan_t5.jsonl; do
    if [ -f "$file" ]; then
        count=$(wc -l < "$file")
        basename=$(basename "$file" _flan_t5.jsonl)
        printf "%-25s: %10s recipes\n" "$basename" "$count"
    fi
done

total_count=$(wc -l < "$OUTPUT_DIR/combined_all_datasets_flan_t5.jsonl")
echo "================================="
printf "%-25s: %10s recipes\n" "TOTAL COMBINED" "$total_count"
echo "================================="

echo ""
echo "🎉 Batch validation completed!"
echo "📁 All files saved in: $OUTPUT_DIR/"
echo "🎯 Ready for training: $OUTPUT_DIR/combined_all_datasets_flan_t5.jsonl"