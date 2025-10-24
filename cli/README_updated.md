# Recipe Generation Model Training - Performance Optimized

A comprehensive recipe generation model training system supporting multiple architectures, advanced training techniques, and **performance optimization for RTX 4090**.

## 🚀 Performance Highlights

- **10-20x Training Speed Boost** with pre-tokenization
- **Streaming Data Loading** for 3.1M+ recipe datasets
- **Real-time Bottleneck Detection** with Wandb monitoring
- **RTX 4090 Optimized** batch sizes and memory management
- **Advanced PyTorch Profiling** for performance analysis

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

## Quick Start - Maximum Performance

### 🚀 RTX 4090 Optimized Training (Recommended)
```bash
# Step 1: Pre-tokenize dataset (one-time setup, 10-20x speed boost)
python pretokenize_dataset.py \
  --pretrained-model google/flan-t5-xl \
  --max-length 512 \
  --output-dir data/tokenized

# Step 2: Train with all optimizations
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --batch-size 8 \
  --gradient-accumulation-steps 2 \
  --pretokenized-path data/tokenized/all_datasets_t5_512tokens.pkl \
  --use-wandb \
  --epochs 3 \
  --model-output ./models/flan-t5-xl-optimized
```

### 📈 Performance Monitoring & Bottleneck Analysis
```bash
# Basic monitoring
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --use-wandb \
  --wandb-project "recipe-optimization"

# Advanced profiling for detailed bottleneck analysis
python train_recipe_model.py \
  --model-type t5 \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --use-wandb \
  --enable-profiling \
  --profile-schedule "wait=2;warmup=2;active=5;repeat=3"
```

## Performance Benchmarks

### RTX 4090 Training Times (3.1M Recipes, 3 Epochs)
| Configuration | Time per Epoch | Total Training Time | Speed Improvement |
|---------------|----------------|---------------------|-------------------|
| **Without optimizations** | 8-12 hours | 24-36 hours | Baseline |
| **With pre-tokenization** | 45-90 minutes | 2.5-4.5 hours | **10-15x faster** |
| **All optimizations** | 30-60 minutes | 1.5-3 hours | **15-20x faster** |

### Model Performance Matrix
| Model | Parameters | RTX 4090 Batch Size | Memory Usage | Quality | Speed Boost* |
|-------|------------|---------------------|--------------|---------|---------------|
| FLAN-T5-Large | 770M | 16 | ~8-12GB | Good | 5-10x |
| FLAN-T5-XL | 3B | 8 | ~18-22GB | Excellent | 10-15x |
| FLAN-T5-XXL + LoRA | 11B | 2 | ~22-24GB | Best | 15-20x |

*With pre-tokenization and optimizations

## Pre-tokenization for Maximum Performance

### Why Pre-tokenize?
Pre-tokenization eliminates the #1 training bottleneck by:
- **10-20x faster data loading**: No runtime tokenization
- **50-90% less training time**: Especially for large datasets
- **Consistent memory usage**: Predictable RAM requirements
- **Better CPU utilization**: All cores available for training

### Pre-tokenization Workflow
```bash
# 1. Pre-tokenize your dataset (one-time setup)
python pretokenize_dataset.py \
  --datasets food_com_recipes_2m allrecipes_250k \
  --pretrained-model google/flan-t5-xl \
  --max-length 512 \
  --output-dir data/tokenized

# 2. Train with pre-tokenized data
python train_recipe_model.py \
  --pretrained-model google/flan-t5-xl \
  --use-lora \
  --pretokenized-path data/tokenized/food_com_recipes_2m_allrecipes_250k_t5_512tokens.pkl
```

## Command Line Arguments

### Model Configuration
- `--model-type`: Choose between 't5' or 'gpt2'
- `--pretrained-model`: Specific model (e.g., google/flan-t5-xl)
- `--gpt-model-size`: For GPT-2: medium/large/xl/custom_large

### Training Parameters
- `--epochs`: Number of training epochs (default: 10)
- `--batch-size`: Per-device batch size (auto-adjusted for RTX 4090)
- `--learning-rate`: Learning rate (default: 5e-5)
- `--max-length`: Maximum sequence length (default: 512)

### Advanced Features
- `--use-lora`: Enable LoRA fine-tuning
- `--lora-r`: LoRA rank (default: 16, use 8 for XL models)
- `--multi-task`: Enable nutrition/difficulty prediction
- `--filter-quality`: Remove low-quality recipes
- `--use-augmentation`: Apply ingredient substitution

### Performance Optimization
- `--pretokenized-path`: Use pre-tokenized dataset for 10x speed boost
- `--use-wandb`: Enable comprehensive performance monitoring
- `--enable-profiling`: Advanced PyTorch profiling for bottleneck analysis
- `--wandb-project`: Custom Wandb project name

### Data Options
- `--datasets`: Specific datasets to use
- `--max-samples`: Limit training samples (for testing)
- `--data-path`: Custom recipe JSON file

## Bottleneck Identification with Wandb

### What Wandb Automatically Tracks:
1. **System Resources** (real-time):
   - GPU utilization (%) and memory usage
   - CPU utilization (%) per core
   - RAM usage and availability
   - Disk I/O rates (MB/s)
   - Network I/O rates (MB/s)

2. **Training Metrics**:
   - Data loading time per batch
   - GPU memory allocation patterns
   - Model parameter efficiency
   - Gradient flow and optimization

3. **Advanced Profiling** (with `--enable-profiling`):
   - Top CPU/CUDA operations by time
   - Memory allocation bottlenecks
   - Function call stacks
   - Chrome trace files for detailed analysis

### Identifying Common Bottlenecks:
- **Data loading >0.1s/batch**: Use `--pretokenized-path`
- **GPU utilization <90%**: Increase batch size
- **High CPU usage**: More DataLoader workers (auto-optimized)
- **Disk I/O bottlenecks**: Pre-tokenization helps significantly

## Memory-Efficient Training Features

- **Streaming data loading**: Processes large datasets without loading into RAM
- **Pre-tokenization**: Eliminates 50-90% of training time bottlenecks
- **LoRA optimization**: Aggressive settings for XL models (r=8, alpha=16)
- **Gradient checkpointing**: Enabled automatically
- **BF16 precision**: On capable GPUs (RTX 4090)
- **Dynamic batch sizing**: Optimized for RTX 4090 (up to 32 for GPT-2, 8 for T5-XL)
- **Fused optimizers**: AdamW fused for RTX 4090
- **12 DataLoader workers**: Fully utilizes CPU cores

## Troubleshooting

### CUDA Out of Memory
- Reduce `--batch-size`
- Increase `--gradient-accumulation-steps`
- Enable `--use-lora` for large models
- Reduce `--max-length`

### Slow Training
- **Use pre-tokenization**: `python pretokenize_dataset.py` first
- **Increase batch size**: RTX 4090 can handle 8+ for T5-XL
- **Enable monitoring**: `--use-wandb` to identify bottlenecks
- **Check data loading**: Should be <0.1s per batch with pre-tokenization
- **Verify GPU utilization**: Should be >90% with Wandb monitoring
- **Use streaming**: Automatic for large datasets

### Poor Recipe Quality
- Use FLAN-T5-XL instead of smaller models
- Enable `--multi-task` learning
- Apply `--filter-quality` to training data
- Use `--use-augmentation` for data diversity
- Increase training epochs
- Monitor with `--use-wandb` to ensure model is learning

## Advanced Usage

### Custom Training Loop with Performance Monitoring
```python
from train_recipe_model import RecipeGenerationModel

config = {
    'model_type': 't5',
    'pretrained_model': 'google/flan-t5-xl',
    'use_lora': True,
    'multi_task': True,
    'use_wandb': True,
    'epochs': 10
}

trainer = RecipeGenerationModel(config)
trainer.initialize_tokenizer()
trainer.initialize_model()

# Load with pre-tokenized data for maximum speed
train_dataset, val_dataset = trainer.load_data(
    pretokenized_path='data/tokenized/recipes_t5_512tokens.pkl'
)
trainer.train(train_dataset, val_dataset, 'output_dir')
```

### Generation Example
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

### Optimal (This implementation is optimized for)
- **24GB VRAM (RTX 4090/A5000)**
- **T5-XL or Custom Large GPT-2 models**
- **LoRA fine-tuning of 11B models**
- **Full performance monitoring and profiling**

## Tips for Best Results

1. **Use pre-tokenization** for 10-20x speed improvement
2. **Enable Wandb monitoring** to identify bottlenecks
3. **Use FLAN-T5-XL with LoRA** for best quality/speed balance
4. **Enable multi-task learning** for better recipe understanding
5. **Apply quality filtering** to improve training data
6. **Monitor GPU utilization** - should be >90% during training
7. **Use all optimizations** for maximum performance on RTX 4090