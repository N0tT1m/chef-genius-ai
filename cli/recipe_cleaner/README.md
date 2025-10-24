# Recipe Dataset Cleaner 🧹

High-performance Rust tool for cleaning millions of recipe records. Removes garbage data, fixes formatting issues, and validates recipe quality.

## Features

- **Blazing Fast**: Parallel processing with Rayon - handles millions of recipes in minutes
- **Smart Detection**: Removes garbage like `:)))`, `lol`, test data, spam, profanity
- **Excel Fix**: Automatically fixes Excel date conversion bugs (`2-Jan` → `1/2`)
- **Deduplication**: Removes duplicate recipes based on title + ingredients fingerprint
- **Format Support**: JSON, JSONL, and CSV files
- **Progress Bars**: Visual progress tracking with indicatif
- **Detailed Stats**: Comprehensive cleaning statistics

## What Gets Removed

### Garbage Patterns
- Emoticons: `:)))`, `:(((`
- Internet slang: `lol`, `haha`, `wtf`, `omg`
- Test data: `test recipe`, `asdfasdf`
- Profanity
- Excessive punctuation: `!!!`, `???`, `....`
- Long number sequences (phone numbers, etc.)
- Repeated characters: `aaaaaaa`

### Quality Issues
- Missing required fields (title, ingredients, or instructions)
- Too few ingredients (< 2 by default)
- Instructions too short (< 20 chars by default)
- Excessive special characters (> 30% by default)
- Excessive capitalization (> 80%)
- Duplicate recipes

### Formatting Fixes
- Excel date conversions: `2-Jan` → `1/2`, `3-Feb` → `3/4`
- Unicode fixes: `°`, `½`, `¼`, `¾`
- Whitespace normalization
- Zero-width character removal

## Build

```bash
cd cli/recipe_cleaner
cargo build --release
```

Or use the build script:
```bash
./build_and_clean.sh
```

## Usage

### Basic Usage

```bash
# Clean a single file
./target/release/recipe_cleaner \
  --input ../../data/training.json \
  --output ../../data/training_cleaned.json

# Clean entire directory recursively
./target/release/recipe_cleaner \
  --input ../../data/datasets/ \
  --output ../../data/cleaned_datasets/ \
  --recursive
```

### Dry Run (Preview)

```bash
# See what would be cleaned without writing files
./target/release/recipe_cleaner \
  --input ../../data/datasets/ \
  --recursive \
  --dry-run \
  --verbose
```

### Advanced Options

```bash
# Stricter quality requirements
./target/release/recipe_cleaner \
  --input ../../data/datasets/ \
  --recursive \
  --min-ingredients 3 \
  --min-instructions-len 50 \
  --max-special-char-ratio 0.2

# Verbose mode (see rejected recipes)
./target/release/recipe_cleaner \
  --input ../../data/datasets/ \
  --recursive \
  --verbose
```

## Command-Line Options

```
Options:
  -i, --input <INPUT>
          Input file or directory

  -o, --output <OUTPUT>
          Output file or directory

  -r, --recursive
          Process directories recursively

      --dry-run
          Dry run - don't write files

  -v, --verbose
          Verbose output

      --min-ingredients <MIN_INGREDIENTS>
          Minimum ingredients count [default: 2]

      --min-instructions-len <MIN_INSTRUCTIONS_LEN>
          Minimum instructions length [default: 20]

      --max-special-char-ratio <MAX_SPECIAL_CHAR_RATIO>
          Maximum special character ratio [default: 0.3]

  -h, --help
          Print help
```

## Performance

**Benchmarks** (on Ryzen 9 7950X):

- **2M recipes (PP_recipes.csv)**: ~2-3 minutes
- **JSON files**: ~10-50K recipes/second
- **CSV files**: ~5-20K recipes/second (parallel processing)

Memory usage: ~100-500MB depending on file size

## Output

The tool generates:

1. **Cleaned files**: Same format as input, with `_cleaned` suffix
2. **Statistics report**:
   ```
   📊 CLEANING STATISTICS
   ====================================
   Total recipes processed:        2,000,000
   Kept (clean):                   1,750,000 ✅
   Fixed formatting issues:          125,000 🔧
   Removed (garbage):                 50,000 🗑️
   Removed (too short):               30,000 📏
   Removed (missing fields):          20,000 ❓
   Removed (duplicates):             150,000 🔁

   Quality rate: 87.5% kept, 12.5% removed
   ```

## Integration with Training

After cleaning, update your training script to use the cleaned datasets:

```python
# In unified_dataset_loader.py or training script
datasets_path = "data/cleaned_datasets"  # Use cleaned version
```

Or clean datasets in-place:
```bash
# Back up first!
cp -r data/datasets data/datasets_backup

# Clean in-place
./target/release/recipe_cleaner \
  --input ../../data/datasets/ \
  --recursive
```

## Field Name Support

The cleaner recognizes various field name formats:

- **Title**: `title`, `name`, `recipe_name`, `RecipeName`, `Title`
- **Ingredients**: `ingredients`, `Ingredients`, `ingredient_tokens`
- **Instructions**: `instructions`, `directions`, `steps`, `Instructions`

## Examples

### Example 1: Quick Test

```bash
# Dry run on one dataset to see stats
./target/release/recipe_cleaner \
  --input ../../data/datasets/indian_food_6k/IndianFoodDatasetCSV.csv \
  --dry-run
```

### Example 2: Clean Everything

```bash
# Clean all datasets, save to new directory
./target/release/recipe_cleaner \
  --input ../../data/datasets/ \
  --output ../../data/cleaned_datasets/ \
  --recursive
```

### Example 3: Very Strict Quality

```bash
# Only keep highest quality recipes
./target/release/recipe_cleaner \
  --input ../../data/datasets/ \
  --recursive \
  --min-ingredients 5 \
  --min-instructions-len 100 \
  --max-special-char-ratio 0.15
```

## License

Same as parent project
