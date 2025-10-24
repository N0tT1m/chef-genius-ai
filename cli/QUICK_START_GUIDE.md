# Quick Start Guide - Improved Recipe Generator

## The Problem You're Experiencing

**Issue:** All generated recipes look the same
**Cause:** Checkpoint 12000 is too early in training - the model hasn't learned diversity yet
**Solution:** Use the new improved generator with diversity checking

## Quick Fix

### Option 1: Use a Later Checkpoint (Best Solution)
```bash
# Find your latest checkpoint
ls -lt checkpoints/

# Use checkpoint 50000 or later
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoints/checkpoint-50000 \
  --check-diversity \
  --output results
```

### Option 2: Use Improved Generator with Current Checkpoint
```bash
# The new generator has better diversity controls
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoints/checkpoint-12000 \
  --enable-beam-search \
  --quality-threshold 0.5 \
  --max-retries 3 \
  --check-diversity \
  --output results_checkpoint_12000
```

## What to Expect at Different Checkpoints

| Checkpoint | Quality | Diversity | Recommendation |
|-----------|---------|-----------|----------------|
| 1-10k | Poor | Very Low | ❌ Don't use - too early |
| 12k-20k | Basic | Low | ⚠️ Limited diversity, use with creative mode |
| 20k-40k | Good | Moderate | ✅ Decent for testing |
| 40k-60k | Very Good | High | ✅ Good for production |
| 60k+ | Excellent | Very High | ✅ Best results |

## Installation

```bash
cd cli
pip install -r requirements_recipes.txt
```

## Basic Usage

```bash
# Simplest usage
python improved_pork_belly_generator.py \
  --checkpoint /path/to/checkpoint \
  --output my_recipes

# With Discord notifications
python improved_pork_belly_generator.py \
  --checkpoint /path/to/checkpoint \
  --discord-webhook "YOUR_WEBHOOK_URL" \
  --output my_recipes

# With diversity check
python improved_pork_belly_generator.py \
  --checkpoint /path/to/checkpoint \
  --check-diversity \
  --output my_recipes
```

## Understanding the Output

### Diversity Check Results

When you run with `--check-diversity`, you'll see:

```
RECIPE DIVERSITY ANALYSIS
================================================================================

📊 Diversity Metrics:
   Average Similarity: 0.850
   Average Lexical Diversity: 0.432
   Unique Ingredients: 12
   Total Recipes: 20

❌ CRITICAL: Recipes are nearly identical. Model checkpoint likely too early in training.

💡 RECOMMENDATIONS:
   1. Use a later checkpoint (checkpoint 12000 may be too early)
   2. Increase generation temperature (try 0.9-1.2)
   3. Enable beam search with higher diversity penalty
   4. Use 'creative' mode instead of 'greedy' mode
   5. Check if model has trained on diverse enough data
```

### What the Numbers Mean

**Average Similarity (BLEU Score)**
- 0.9+ = Recipes are almost identical ❌
- 0.7-0.9 = Recipes are very similar ⚠️
- 0.5-0.7 = Some variety 🆗
- <0.5 = Good diversity ✅

**Unique Ingredients**
- Should be at least 50% of total recipes
- For 20 recipes, expect 10+ unique ingredient names

**Lexical Diversity**
- 0.0-0.3 = Repetitive ❌
- 0.3-0.5 = Moderate variety 🆗
- 0.5+ = Good variety ✅

## Comparing Old vs New Generator

### Old Generator (pork_belly_recipe_generator.py)
```bash
python pork_belly_recipe_generator.py \
  --checkpoint ./checkpoint \
  --discord-webhook "URL" \
  --output results.json
```

**Limitations:**
- No diversity checking
- No retry logic
- Limited export formats
- No validation
- No few-shot learning

### New Generator (improved_pork_belly_generator.py)
```bash
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoint \
  --discord-webhook "URL" \
  --check-diversity \
  --enable-beam-search \
  --export-formats all \
  --output results
```

**Advantages:**
- ✅ Automatic diversity checking
- ✅ Retry logic with quality thresholds
- ✅ 5 export formats (MD, HTML, PDF, TXT, JSON)
- ✅ Ingredient/instruction validation
- ✅ Few-shot learning
- ✅ BLEU/ROUGE/perplexity metrics
- ✅ Real-time Discord updates
- ✅ Progress bars
- ✅ Memory optimization

## Export Formats

### All Formats at Once
```bash
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoint \
  --export-formats all \
  --output my_recipes
```

Generates:
- `my_recipes.md` - Markdown with tables
- `my_recipes.html` - Beautiful web page
- `my_recipes.pdf` - Print-ready PDF
- `my_recipes.txt` - Plain text
- `my_recipes.json` - Full data with metrics

### Specific Formats
```bash
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoint \
  --export-formats markdown html \
  --output my_recipes
```

## Testing Before Full Run

### Quick Test (5 Recipes)
Edit the generator to use fewer recipes:

```python
# In improved_pork_belly_generator.py
def _create_prompts(self):
    prompts = []
    # Only add 5 prompts for testing
    prompts.extend([
        RecipePrompt(...),  # Just 5 recipes
        RecipePrompt(...),
        RecipePrompt(...),
        RecipePrompt(...),
        RecipePrompt(...),
    ])
    return prompts
```

Then run:
```bash
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoint \
  --check-diversity \
  --output test_run
```

## Troubleshooting Common Issues

### Issue: "All recipes are the same"
**Diagnosis:**
```bash
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoint \
  --check-diversity \
  --output diagnosis
```

**Solutions (in order of effectiveness):**
1. Use checkpoint 50000+ instead of 12000
2. Use only creative mode (edit config in code)
3. Enable beam search
4. Increase temperature to 1.0-1.2 (edit config in code)

### Issue: "Quality scores are low"
**Solutions:**
- Lower quality threshold: `--quality-threshold 0.4`
- Increase retries: `--max-retries 5`
- Use few-shot learning (enabled by default)

### Issue: "Generation is too slow"
**Solutions:**
- Reduce parallel workers: `--parallel 1`
- Use greedy mode only (edit config)
- Disable beam search (don't use `--enable-beam-search`)

### Issue: "Out of memory"
**Solutions:**
- Reduce parallel workers: `--parallel 1`
- Generate fewer recipes at a time
- Use smaller checkpoint/model

## Running Tests

```bash
# Run all unit tests
python test_recipe_generator.py

# Test just diversity metrics
python -c "
from test_recipe_generator import TestDiversityAnalyzer
recipes = ['recipe 1 text', 'recipe 2 text', ...]
result = TestDiversityAnalyzer.analyze_recipe_diversity(recipes)
print(result)
"
```

## Discord Webhook Setup

1. Go to your Discord server
2. Server Settings → Integrations → Webhooks
3. Create webhook, copy URL
4. Use in command:

```bash
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoint \
  --discord-webhook "https://discord.com/api/webhooks/YOUR_WEBHOOK_URL" \
  --output results
```

You'll receive:
- Session start notification
- Progress updates every 5 recipes
- Category completion notifications
- High-quality recipe previews
- Final comprehensive report
- Diversity analysis (if enabled)

## Example: Complete Workflow

```bash
# 1. Install dependencies
pip install -r requirements_recipes.txt

# 2. Find your best checkpoint
ls -lt checkpoints/ | head -10

# 3. Test with diversity check
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoints/checkpoint-50000 \
  --check-diversity \
  --output test_run \
  --export-formats markdown

# 4. Review test_run.md and diversity report

# 5. If diversity is good, run full generation with all features
python improved_pork_belly_generator.py \
  --checkpoint ./checkpoints/checkpoint-50000 \
  --discord-webhook "YOUR_WEBHOOK" \
  --enable-beam-search \
  --quality-threshold 0.7 \
  --max-retries 3 \
  --export-formats all \
  --check-diversity \
  --output final_recipes

# 6. Check outputs
ls -lh final_recipes.*
# final_recipes.md
# final_recipes.html
# final_recipes.pdf
# final_recipes.txt
# final_recipes.json
```

## Next Steps

1. **If diversity is still low:** Train model longer or increase dataset diversity
2. **If quality is good:** Use for production recipe generation
3. **If specific categories fail:** Adjust prompts for those categories
4. **For custom recipes:** Create your own generator extending `BaseRecipeGenerator`

## Getting Help

Check these in order:
1. Read the error message
2. Check diversity analysis output
3. Review RECIPE_GENERATOR_README.md
4. Run tests: `python test_recipe_generator.py`
5. Check Discord notifications for hints

## Summary Commands

```bash
# Quick test
python improved_pork_belly_generator.py --checkpoint ./checkpoint --check-diversity --output test

# Production run
python improved_pork_belly_generator.py --checkpoint ./checkpoint --discord-webhook "URL" --export-formats all --check-diversity --output prod

# Debug diversity
python improved_pork_belly_generator.py --checkpoint ./checkpoint --check-diversity --max-retries 5 --output debug
```

Remember: **Checkpoint 12000 is very early. For best results, use checkpoint 50000+**
