# Improved Recipe Generator System

A comprehensive, modular recipe generation system with advanced features for quality, diversity, and export capabilities.

## New Features

### 1. **Base Recipe Generator Class**
- Eliminates code duplication across specialized generators
- Shared functionality for all recipe types
- Easy to extend for new recipe categories

### 2. **Advanced Generation Features**
- **Few-Shot Learning**: Includes example recipes in prompts for better quality
- **Beam Search**: Optional beam search for more diverse outputs
- **Retry Logic**: Automatically retries failed generations with quality thresholds
- **Multiple Generation Modes**: Greedy, Normal, and Creative modes
- **Prompt Caching**: Caches tokenized prompts for faster generation
- **Memory Optimization**: Automatic cleanup and garbage collection

### 3. **Quality Metrics**
- **BLEU Score**: Measures n-gram overlap with reference recipes
- **ROUGE Scores**: ROUGE-1, ROUGE-2, ROUGE-L for comprehensive comparison
- **Perplexity**: Model confidence scoring
- **Coherence Checking**: Validates ingredient-instruction alignment
- **Lexical Diversity**: Measures vocabulary richness
- **Readability Metrics**: Flesch Reading Ease scoring
- **Completeness Analysis**: Checks for required recipe sections

### 4. **Validation System**
- **Cooking Time Validation**: Flags unrealistic cooking times
- **Temperature Validation**: Ensures safe cooking temperatures
- **Quantity Validation**: Detects unusual ingredient amounts
- **Ingredient-Instruction Alignment**: Verifies all ingredients are used

### 5. **Export Formats**
- **Markdown**: Clean, portable format
- **HTML**: Beautiful web-ready format with CSS styling
- **PDF**: Print-ready documents (requires reportlab)
- **Plain Text**: Simple text format
- **JSON**: Structured data with all metrics

### 6. **Enhanced Discord Integration**
- **Session Start Notifications**: Announces generation sessions
- **Real-time Progress Updates**: Shows progress during generation
- **Category Completion**: Reports when categories finish
- **Recipe Previews**: Sends high-quality recipe previews
- **Comprehensive Final Report**: Detailed statistics and top recipes
- **Error Notifications**: Alerts on failures

### 7. **Diversity Analysis**
- **Similarity Detection**: Detects when all recipes are too similar
- **Unique Ingredient Tracking**: Counts variety across recipes
- **Early Checkpoint Warning**: Flags when checkpoint is too early
- **Recommendations**: Suggests fixes for low diversity

### 8. **Pluggable Prompts**
- Load prompts from YAML/JSON files
- Save prompts for reuse
- Easy prompt customization without code changes

## Installation

```bash
cd cli
pip install -r requirements.txt
```

### Requirements
```
torch>=2.0.0
transformers>=4.30.0
requests>=2.28.0
pyyaml>=6.0
tqdm>=4.65.0
reportlab>=4.0.0  # Optional, for PDF export
```

## Usage

### Basic Usage

```bash
python improved_pork_belly_generator.py \
  --checkpoint /path/to/checkpoint \
  --output results
```

### With All Features

```bash
python improved_pork_belly_generator.py \
  --checkpoint /path/to/checkpoint \
  --discord-webhook "https://discord.com/api/webhooks/..." \
  --enable-beam-search \
  --quality-threshold 0.7 \
  --max-retries 3 \
  --parallel 2 \
  --export-formats all \
  --check-diversity \
  --output pork_belly_results
```

### Options

- `--checkpoint`: Path to model checkpoint (required)
- `--discord-webhook`: Discord webhook URL for notifications
- `--parallel`: Number of parallel generations (default: 2)
- `--output`: Base filename for output files
- `--enable-beam-search`: Enable beam search on retries
- `--quality-threshold`: Minimum quality score (default: 0.6)
- `--max-retries`: Maximum retry attempts (default: 2)
- `--export-formats`: Export formats (choices: markdown, html, pdf, text, json, all)
- `--prompts-file`: Load prompts from YAML/JSON file
- `--save-prompts`: Save prompts to YAML/JSON file
- `--check-diversity`: Run diversity analysis on results

## Addressing the "All Recipes Are the Same" Issue

If you're seeing identical or very similar recipes, this is likely because **checkpoint 12000 is too early in training**. The model hasn't learned enough diversity yet.

### Solutions:

1. **Use a Later Checkpoint**
   ```bash
   # Instead of checkpoint-12000, use checkpoint-50000 or later
   python improved_pork_belly_generator.py --checkpoint ./checkpoints/checkpoint-50000
   ```

2. **Increase Generation Temperature**
   - Edit the `GenerationConfig` in the code
   - Change temperature from 0.5 to 0.9-1.2 for more diversity

3. **Use Creative Mode Only**
   ```python
   config = GenerationConfig(
       generation_modes=["creative"],  # Skip greedy and normal
       ...
   )
   ```

4. **Enable Beam Search**
   ```bash
   python improved_pork_belly_generator.py \
     --checkpoint /path/to/checkpoint \
     --enable-beam-search
   ```

5. **Run Diversity Check**
   ```bash
   python improved_pork_belly_generator.py \
     --checkpoint /path/to/checkpoint \
     --check-diversity
   ```

   This will automatically analyze diversity and provide specific recommendations.

## Creating Custom Recipe Generators

```python
from base_recipe_generator import BaseRecipeGenerator, RecipePrompt, GenerationConfig

class MyCustomGenerator(BaseRecipeGenerator):
    def get_recipe_type(self) -> str:
        return "Custom Recipe Type"

    def _create_prompts(self) -> List[RecipePrompt]:
        return [
            RecipePrompt(
                name="My Recipe",
                prompt="Create a custom recipe...",
                category="custom",
                difficulty="normal",
                expected_features=["feature1", "feature2"]
            )
        ]

# Use it
config = GenerationConfig(enable_few_shot=True, retry_on_failure=True)
generator = MyCustomGenerator(discord_webhook="...", config=config)
results = generator.generate_all_recipes("./checkpoint")
```

## Running Tests

```bash
# Run unit tests
python test_recipe_generator.py

# Test diversity on existing results
from test_recipe_generator import run_diversity_check
run_diversity_check(results['individual_results'])
```

## Export Examples

### Markdown Export
Clean, portable format with tables and formatting:
```
# Pork Belly Collection

**Success Rate:** 85%
**Average Quality:** 0.756

## Traditional

### Classic Crispy Pork Belly
**Quality:** 0.823 | **Mode:** normal

INGREDIENTS:
- 2-3 lbs pork belly
...
```

### HTML Export
Beautiful web page with CSS styling, color-coded quality scores, and responsive design.

### PDF Export
Professional PDF with formatted tables and sections, ready for printing.

## Metrics Explained

### Quality Score (0.0 - 1.0)
- **0.8+**: Excellent - Complete, coherent, well-structured
- **0.6-0.8**: Good - Minor issues but usable
- **0.4-0.6**: Fair - Needs improvement
- **<0.4**: Poor - Significant issues

### BLEU Score (0.0 - 1.0)
- Measures similarity to reference recipes
- Higher = more similar to reference
- **For diversity check**: Lower scores are better (want variety)

### Coherence Score (0.0 - 1.0)
- Measures ingredient-instruction alignment
- Checks if all ingredients are used
- Validates recipe structure

### Perplexity
- Model's confidence in generation
- Lower = more confident
- **<10**: Very confident
- **10-50**: Normal confidence
- **>50**: Low confidence (may be hallucinating)

## Discord Notification Features

The enhanced Discord integration provides:

1. **Session Start**: Announces when generation begins
2. **Progress Updates**: Every 5 recipes, shows current progress
3. **Category Complete**: Reports when each category finishes
4. **Recipe Previews**: Sends snippets of high-quality recipes (quality > 0.8)
5. **Final Report**: Comprehensive statistics with:
   - Success rate
   - Average quality
   - Performance metrics
   - Top 5 recipes
   - Category breakdown
   - Mode distribution

6. **Diversity Warnings**: Alerts if recipes are too similar

## File Structure

```
cli/
├── base_recipe_generator.py      # Base class for all generators
├── recipe_metrics.py              # BLEU, ROUGE, coherence, etc.
├── recipe_exporters.py            # Export to multiple formats
├── discord_notifier.py            # Enhanced Discord integration
├── test_recipe_generator.py       # Unit tests and diversity check
├── improved_pork_belly_generator.py  # New pork belly generator
├── pork_belly_recipe_generator.py    # Original (for comparison)
└── RECIPE_GENERATOR_README.md     # This file
```

## Troubleshooting

### "Recipes are identical"
- **Cause**: Checkpoint too early in training
- **Fix**: Use checkpoint 50000+, increase temperature, use creative mode

### "Low quality scores"
- **Cause**: Model needs more training or better prompts
- **Fix**: Use later checkpoint, enable few-shot, adjust quality threshold

### "Generation is slow"
- **Fix**: Reduce parallel workers, use greedy mode only, disable beam search

### "Out of memory"
- **Fix**: Reduce parallel workers to 1, clear cache more frequently

### "PDF export fails"
- **Fix**: Install reportlab: `pip install reportlab`

## Best Practices

1. **Always run diversity check** on new checkpoints
2. **Start with checkpoint 50000+** for decent diversity
3. **Use creative mode** for maximum variety
4. **Enable few-shot learning** for better quality
5. **Set quality threshold to 0.6-0.7** for balance
6. **Export to multiple formats** for flexibility
7. **Use Discord notifications** for long runs
8. **Save prompts to files** for reproducibility

## Performance Tips

1. Keep parallel workers at 2-3 maximum
2. Clear cache every 5-10 recipes
3. Use greedy mode first to test checkpoint quality
4. Enable beam search only on retries
5. Use progress bars to monitor long runs

## Future Enhancements

- [ ] Support for reference recipes database
- [ ] Automatic hyperparameter tuning
- [ ] Multi-language recipe generation
- [ ] Image generation integration
- [ ] Recipe difficulty auto-classification
- [ ] Nutritional information extraction
- [ ] Allergen detection
- [ ] Cost estimation

## License

Part of the Chef Genius project.
