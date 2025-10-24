# Recipe Generation Model Training

A comprehensive recipe generation model training system supporting multiple architectures, advanced training techniques, and multi-task learning.

## Features

### 🏗️ Model Architectures
- **T5/FLAN-T5**: Encoder-decoder architecture optimized for structured generation
- **GPT-2**: Decoder-only models with scalable sizes (355M to 2B parameters)
- **Multi-size support**: From efficient models to RTX 4090-optimized large models

### 🚀 Advanced Training Techniques
- **LoRA (Low-Rank Adaptation)**: Efficient fine-tuning of large models
- **Gradient Checkpointing**: Memory-efficient training
- **Mixed Precision**: BF16 support for faster training
- **Dynamic Batch Sizing**: Device-optimized configurations

### 🎯 Multi-Task Learning
- **Recipe Generation**: Primary task
- **Nutrition Classification**: Healthy/Moderate/Indulgent categories
- **Difficulty Prediction**: Easy/Medium/Hard classification

### 📊 Evaluation Metrics
- **ROUGE Scores**: Text generation quality
- **BLEU Score**: Translation-style metrics
- **Ingredient Coverage**: Recipe-specific accuracy
- **Structure Score**: Recipe format compliance

### 🔧 Data Processing
- **Streaming Data Loading**: Memory-efficient processing of large datasets
- **Pre-tokenization**: Eliminate runtime tokenization bottlenecks
- **Quality Filtering**: Remove low-quality recipes
- **Data Augmentation**: Ingredient substitution for diversity
- **Multi-dataset Support**: Combine multiple recipe datasets

### 📈 Performance Monitoring
- **Weights & Biases Integration**: Real-time system monitoring
- **Advanced PyTorch Profiling**: Detailed bottleneck identification
- **GPU/CPU/Memory Tracking**: Comprehensive resource utilization
- **I/O Monitoring**: Data loading and disk performance analysis

## Installation

```bash
# Required dependencies
pip install torch>=2.7.1
pip install transformers>=4.52.0
pip install datasets>=2.12.0

# Optional for advanced features
pip install peft  # For LoRA
pip install rouge-score nltk  # For evaluation metrics
pip install wandb  # For performance monitoring
pip install psutil gputil  # For system monitoring
```

## Usage

### Basic T5 Training (Recommended)
```bash
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-large \
  --model-output ./models/flan-t5-recipes \
  --epochs 10 \
  --batch-size 8
```

### Advanced T5 with LoRA (3B+ models on RTX 4090)
```bash
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --lora-r 8 \
  --model-output ./models/flan-t5-xl-lora \
  --batch-size 8 \
  --gradient-accumulation-steps 2
```

### Pre-tokenization for 10x Faster Training
```bash
# First, pre-tokenize your dataset
python pretokenize_dataset.py \
  --pretrained-model google/flan-t5-xl \
  --max-length 512 \
  --output-dir data/tokenized

# Then train with pre-tokenized data
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --pretokenized-path data/tokenized/all_datasets_t5_512tokens.pkl \
  --model-output ./models/flan-t5-xl-fast
```

### Performance Monitoring with Wandb
```bash
# Basic monitoring
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --use-wandb \
  --wandb-project "recipe-optimization"

# Advanced profiling for bottleneck analysis
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --use-wandb \
  --enable-profiling \
  --profile-schedule "wait=2;warmup=2;active=5;repeat=3"
```

### Multi-Task Learning
```bash
python train_recipe_model.py \
  --model-type t5 \
  --multi-task \
  --filter-quality \
  --use-augmentation \
  --model-output ./models/multitask-recipes
```

### Large GPT-2 Training
```bash
python train_recipe_model.py \
  --model-type gpt2 \
  --gpt-model-size xl \
  --model-output ./models/gpt2-xl-recipes \
  --batch-size 4 \
  --gradient-accumulation-steps 8
```

## Model Options

### T5/FLAN-T5 Models
| Model | Parameters | RTX 4090 Batch Size | Memory Usage | Quality |
|-------|------------|---------------------|--------------|---------|
| FLAN-T5-Large | 770M | 8-16 | ~8-12GB | Good |
| FLAN-T5-XL | 3B | 2-4 | ~18-22GB | Excellent |
| FLAN-T5-XXL + LoRA | 11B | 1 | ~22-24GB | Best |

### GPT-2 Models
| Size | Parameters | RTX 4090 Batch Size | Memory Usage |
|------|------------|---------------------|--------------|
| Medium | 355M | 16 | ~6GB |
| Large | 762M | 8 | ~12GB |
| XL | 1.5B | 4 | ~18GB |
| Custom Large | 2B | 2-4 | ~20GB |

## Command Line Arguments

### Model Configuration
- `--model-type`: Choose between 't5' or 'gpt2'
- `--pretrained-model`: Specific model (e.g., google/flan-t5-xl)
- `--gpt-model-size`: For GPT-2: medium/large/xl/custom_large

### Training Parameters
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Per-device batch size (auto-adjusted)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--max-length`: Maximum sequence length (default: 512)

### Advanced Features
- `--use-lora`: Enable LoRA fine-tuning
- `--lora-r`: LoRA rank (default: 16)
- `--multi-task`: Enable nutrition/difficulty prediction
- `--filter-quality`: Remove low-quality recipes
- `--use-augmentation`: Apply ingredient substitution

### Data Options
- `--datasets`: Specific datasets to use
- `--max-samples`: Limit training samples (for testing)
- `--data-path`: Custom recipe JSON file

## Performance Optimization

### For RTX 4090 (24GB VRAM)
```bash
# Maximum T5 model with LoRA
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --batch-size 2 \
  --gradient-accumulation-steps 16 \
  --max-length 1024

# Large GPT-2 for comparison
python train_recipe_model.py \
  --model-type gpt2 \
  --gpt-model-size custom_large \
  --batch-size 4 \
  --gradient-accumulation-steps 8
```

### Memory-Efficient Training
- Use LoRA for large models: `--use-lora`
- Gradient checkpointing is enabled automatically
- BF16 precision on capable GPUs
- Dynamic batch size adjustment

## Dataset Format

The system expects recipes in this JSON format:
```json
{
  "title": "Recipe Name",
  "cuisine": "Italian",
  "cooking_time": "30 minutes",
  "servings": 4,
  "ingredients": ["ingredient1", "ingredient2"],
  "instructions": ["step1", "step2"]
}
```

## Evaluation

The system automatically evaluates models using:
- **ROUGE-1/2/L**: Text generation quality
- **BLEU Score**: Sequence similarity
- **Ingredient Coverage**: Recipe-specific accuracy
- **Structure Score**: Format compliance

## Advanced Usage

### Custom Training Loop
```python
from train_recipe_model import RecipeGenerationModel

config = {
    'model_type': 't5',
    'use_lora': True,
    'multi_task': True,
    'epochs': 10
}

trainer = RecipeGenerationModel(config)
trainer.initialize_tokenizer()
trainer.initialize_model()

train_dataset, val_dataset = trainer.load_data()
trainer.train(train_dataset, val_dataset, 'output_dir')
```

### Generation
```python
# Generate a recipe
recipe = trainer.generate_sample(
    "Generate recipe for: Pasta Carbonara using ingredients: pasta, eggs, bacon, cheese"
)
print(recipe)
```

## Hardware Requirements

### Minimum
- 8GB VRAM (GTX 1080/RTX 2070)
- T5-Base or GPT-2 Medium models

### Recommended
- 16GB VRAM (RTX 3080/4070)
- T5-Large or GPT-2 Large models

### Optimal
- 24GB VRAM (RTX 4090/A5000)
- T5-XL or Custom Large GPT-2 models
- LoRA fine-tuning of 11B models

## Tips for Best Results

1. **Use T5/FLAN-T5** for best recipe generation quality
2. **Enable multi-task learning** for better recipe understanding
3. **Apply quality filtering** to improve training data
4. **Use LoRA** to train larger models efficiently
5. **Set appropriate batch sizes** based on your GPU memory

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size`
- Increase `--gradient-accumulation-steps`
- Enable `--use-lora` for large models
- Reduce `--max-length`

### Poor Recipe Quality
- Use T5 instead of GPT-2
- Enable `--multi-task` learning
- Apply `--filter-quality` to training data
- Increase training epochs

### Slow Training
- Increase `--batch-size` if memory allows
- Use BF16 precision (automatic on supported GPUs)
- Enable gradient checkpointing (automatic)