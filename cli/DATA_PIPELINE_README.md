# 🍳 Chef Genius Data Pipeline

## Overview
This directory contains the enterprise-grade data validation and training pipeline for Chef Genius. The pipeline automatically processes raw CSV datasets into clean, high-quality FLAN-T5 training data.

## 🚨 Important: Generated Files Not in Git
The following files are **automatically generated** and **not stored in git** due to size (2.49M+ records):

```
validated_datasets/          # Generated validation output
├── combined_all_datasets_flan_t5.jsonl    # 2.49M training examples
├── recipe_nlg_flan_t5.jsonl               # 2.23M examples  
├── food_com_raw_flan_t5.jsonl             # 231K examples
├── recipe_images_flan_t5.jsonl            # 13K examples
└── *.jsonl                                # Other validated datasets

*.jsonl                      # Individual JSONL files
flan_t5_*.jsonl             # FLAN-T5 training files
validated_*.jsonl           # Validation output files
```

## 🚀 Automatic Pipeline
When you run training, the pipeline **automatically**:

1. **Checks** if validated data exists
2. **Generates** clean JSONL data if missing (via Rust validator)
3. **Loads** high-quality training examples (quality ≥ 0.6)
4. **Starts** training with clean data

## 📊 Data Sources (Raw CSV)
These CSV files **are** in git and used as input:

```
data/datasets/
├── recipe_nlg/RecipeNLG_dataset.csv           # 2.2M recipes
├── food_com_recipes_2m/RAW_recipes.csv        # 267K recipes  
├── indian_food_6k/IndianFoodDatasetCSV.csv    # 6K recipes
├── recipe_ingredient_images/*.csv             # 13K recipes
└── */dataset.csv                              # Various other datasets
```

## 🛠️ Manual Data Generation
If you need to regenerate the validated data:

```bash
# Generate all validated datasets
./batch_validate_all_datasets.sh

# Or validate individual datasets
./recipe_data_validator/target/release/validate_recipes \
    -i data/datasets/recipe_nlg/RecipeNLG_dataset.csv \
    -f recipe_nlg \
    -o validated_datasets/recipe_nlg_validated.jsonl \
    -t validated_datasets/recipe_nlg_flan_t5.jsonl
```

## 🎯 Training Integration
The training script automatically handles data:

```python
# In complete_optimized_training.py
def create_unified_dataloader(self):
    # Automatically generates validated data if missing
    # Uses clean JSONL format instead of broken CSV parsing
    return optimized_dataloader
```

## 📈 Quality Metrics
- **Total Training Examples**: 2,490,151
- **Quality Threshold**: ≥ 0.6 (high quality only)
- **Format**: FLAN-T5 optimized input-output pairs
- **Deduplication**: SHA256 hash-based
- **Validation**: Rust-powered enterprise pipeline

## 🔧 Architecture

```
Raw CSV Files → Rust Validator → Quality Filter → FLAN-T5 Format → Training
     (git)         (fast)         (score≥0.6)      (clean)       (auto)
```

## 🚫 What NOT to Commit
Never commit these large generated files:
- `validated_datasets/` directory
- `*.jsonl` files
- `flan_t5_*.jsonl` files
- `validated_*.jsonl` files

They're automatically generated as needed and can be 1GB+ in size.

## ✅ First-Time Setup
1. Clone repo (CSV data included)
2. Build Rust validator: `cd recipe_data_validator && cargo build --release`
3. Run training: `python3 complete_optimized_training.py`
4. Pipeline automatically generates clean data and starts training

The broken CSV parsing that caused echo responses is completely replaced with this clean pipeline! 🎉