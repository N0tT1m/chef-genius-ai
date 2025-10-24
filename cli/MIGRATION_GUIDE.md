# Migration Guide: V1 → V2 Training System

Complete guide for migrating from the old monolithic training system to the new modular architecture.

## Overview

The V2 training system is a complete rewrite with:
- **Modular architecture** instead of monolithic 1283-line trainer
- **Type-safe YAML configs** instead of hardcoded parameters
- **Validation support** with comprehensive metrics
- **LoRA support** for efficient large model training
- **Callback system** for extensibility

## Breaking Changes

### 1. Import Changes

**Old:**
```python
from complete_optimized_training import CompleteOptimizedTrainer
from ryzen_4090_optimized_training import Ryzen4090OptimizedConfig
```

**New:**
```python
from training import ModularTrainer, CompleteTrainingConfig, create_default_config
```

### 2. Trainer Initialization

**Old:**
```python
trainer = CompleteOptimizedTrainer(
    model=model,
    tokenizer=tokenizer,
    output_dir="./output",
    batch_size=32,
    gradient_accumulation_steps=1,
    discord_webhook="https://...",
    alert_phone="+1234567890",
    wandb_project="chef-genius",
    use_wandb=True,
    enable_mixed_precision=False,
    disable_compilation=False,
    disable_cudagraphs=True,
    dataloader_num_workers=8
)
```

**New:**
```python
# Option A: YAML config (recommended)
config = CompleteTrainingConfig.from_yaml("configs/default_config.yaml")
config.monitoring.discord_webhook = "https://..."
config.monitoring.alert_phone = "+1234567890"

trainer = ModularTrainer(model, tokenizer, config)

# Option B: Programmatic config
config = create_default_config()
config.training.batch_size = 32
config.monitoring.discord_webhook = "https://..."
config.monitoring.use_wandb = True

trainer = ModularTrainer(model, tokenizer, config)
```

### 3. Training Invocation

**Old:**
```python
trainer.train_complete_optimized(epochs=3)
```

**New:**
```python
# Epochs specified in config
trainer.train()
```

### 4. Configuration Structure

**Old:** Scattered parameters across multiple files and hardcoded values

**New:** Centralized, hierarchical configuration

```yaml
# configs/my_config.yaml
training:
  num_epochs: 3
  batch_size: 32

data:
  min_quality_score: 0.6

optimization:
  learning_rate: 0.0005
```

## Step-by-Step Migration

### Step 1: Install New Dependencies

```bash
pip install -r training_requirements.txt
```

Key new dependencies:
- `pydantic>=2.0.0` - Configuration validation
- `rouge-score>=0.1.2` - ROUGE metrics
- `nltk>=3.8` - BLEU metrics
- `peft>=0.5.0` - LoRA support

### Step 2: Create Configuration File

Start with the default config:

```bash
cp configs/default_config.yaml configs/my_training.yaml
```

Edit `configs/my_training.yaml` to match your old settings:

```yaml
# Map your old parameters to new config
experiment_name: "my_experiment"  # NEW: name your experiment

# Old: batch_size=32
training:
  batch_size: 32

# Old: gradient_accumulation_steps=1
training:
  gradient_accumulation_steps: 1

# Old: discord_webhook="..."
monitoring:
  discord_webhook: "https://discord.com/api/webhooks/..."

# Old: wandb_project="chef-genius"
monitoring:
  wandb_project: "chef-genius"

# Old: output_dir="./output"
output_dir: "./output"
```

### Step 3: Update Your Training Script

**Old script:**
```python
#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from complete_optimized_training import CompleteOptimizedTrainer
import torch

# Load model
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = AutoModelForSeq2SeqLM.from_pretrained(
    "google/flan-t5-large",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create trainer
trainer = CompleteOptimizedTrainer(
    model=model,
    tokenizer=tokenizer,
    output_dir="./optimized_model",
    batch_size=32,
    discord_webhook=os.environ.get("DISCORD_WEBHOOK"),
    wandb_project="chef-genius-optimized",
    use_wandb=True,
    gradient_accumulation_steps=1,
    enable_mixed_precision=False,
    disable_cudagraphs=True
)

# Train
trainer.train_complete_optimized(epochs=3)
```

**New script:**
```python
#!/usr/bin/env python3
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from training import CompleteTrainingConfig, ModularTrainer
import torch
import os

# Load configuration
config = CompleteTrainingConfig.from_yaml("configs/my_training.yaml")

# Override with environment variables (optional)
if os.environ.get("DISCORD_WEBHOOK"):
    config.monitoring.discord_webhook = os.environ["DISCORD_WEBHOOK"]

# Load model
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(
    config.model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto"
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Create trainer
trainer = ModularTrainer(model, tokenizer, config)

# Train (epochs from config)
trainer.train()
```

### Step 4: Verify Equivalence

Compare old vs new configuration:

| Old Parameter | New Location | Example |
|--------------|--------------|---------|
| `batch_size` | `training.batch_size` | `32` |
| `gradient_accumulation_steps` | `training.gradient_accumulation_steps` | `1` |
| `output_dir` | `output_dir` | `"./output"` |
| `discord_webhook` | `monitoring.discord_webhook` | `"https://..."` |
| `alert_phone` | `monitoring.alert_phone` | `"+1234567890"` |
| `wandb_project` | `monitoring.wandb_project` | `"chef-genius"` |
| `use_wandb` | `monitoring.use_wandb` | `true` |
| `enable_mixed_precision` | `training.use_bf16` | `true` |
| `disable_cudagraphs` | `training.disable_cudagraphs` | `true` |
| `dataloader_num_workers` | `data.num_workers` | `8` |
| `epochs` (arg) | `training.num_epochs` | `3` |

## New Features to Adopt

### 1. Validation Set

The V2 system automatically creates a validation set:

```yaml
data:
  train_split: 0.9  # 90% train, 10% validation
```

Benefits:
- Monitor overfitting
- Early stopping based on validation metrics
- Better model selection

### 2. Comprehensive Metrics

Enable additional metrics beyond loss:

```yaml
evaluation:
  compute_bleu: true
  compute_rouge: true
  compute_perplexity: true
  compute_ingredient_coherence: true
  compute_instruction_quality: true
```

### 3. LoRA for Large Models

If you're training FLAN-T5-XXL (11B), use LoRA:

```yaml
model_name: "google/flan-t5-xxl"

training:
  use_lora: true
  lora_r: 16
  lora_alpha: 32
  batch_size: 16  # Smaller batch needed
  gradient_accumulation_steps: 4  # Effective batch = 64
```

Benefits:
- 10-100x fewer parameters to train
- ~30% less memory usage
- Faster iteration
- Adapter files are only ~10MB

### 4. Early Stopping

Prevent overfitting with automatic early stopping:

```yaml
training:
  early_stopping_patience: 3  # Stop if no improvement for 3 validations
  early_stopping_threshold: 0.0001
```

### 5. YAML Configuration

Version control your experiments:

```bash
git add configs/experiment_v1.yaml
git commit -m "Add experiment v1 config"
```

Track experiments systematically:
```bash
configs/
├── baseline_config.yaml
├── high_lr_experiment.yaml
├── lora_experiment.yaml
└── ablation_study_config.yaml
```

## Common Migration Patterns

### Pattern 1: Command-line Arguments

**Old:**
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--epochs', type=int, default=3)
args = parser.parse_args()

trainer = CompleteOptimizedTrainer(
    # ...
    batch_size=args.batch_size
)
trainer.train_complete_optimized(epochs=args.epochs)
```

**New:**
```python
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='configs/default_config.yaml')
parser.add_argument('--batch-size', type=int)  # Override config
parser.add_argument('--epochs', type=int)
args = parser.parse_args()

config = CompleteTrainingConfig.from_yaml(args.config)
if args.batch_size:
    config.training.batch_size = args.batch_size
if args.epochs:
    config.training.num_epochs = args.epochs

trainer = ModularTrainer(model, tokenizer, config)
trainer.train()
```

### Pattern 2: Docker Training

**Old Dockerfile:**
```dockerfile
CMD python complete_optimized_training.py \
    --pretrained-model google/flan-t5-large \
    --model-output /output \
    --epochs 3 \
    --batch-size 32
```

**New Dockerfile:**
```dockerfile
COPY configs/training_config.yaml /app/config.yaml
CMD python train_v2.py --config /app/config.yaml
```

### Pattern 3: Multiple Experiments

**Old:** Copy/paste script and modify parameters

**New:** Create config files

```bash
# Run multiple experiments
python train_v2.py --config configs/baseline.yaml
python train_v2.py --config configs/high_lr.yaml
python train_v2.py --config configs/lora.yaml
```

## Troubleshooting

### Issue: "Module 'training' not found"

**Solution:** Ensure you're in the correct directory

```bash
cd /path/to/chef-genius/cli
python -c "import training; print(training.__version__)"  # Should print 2.0.0
```

### Issue: "Pydantic validation error"

**Solution:** Check your YAML syntax and types

```python
# Debug configuration
config = CompleteTrainingConfig.from_yaml("configs/my_config.yaml")
config.print_summary()  # See what was loaded
```

### Issue: "Different results than V1"

**Possible causes:**
1. **Validation split** - V2 uses 90/10 split by default
2. **Learning rate schedule** - V2 uses linear warmup by default
3. **Random seed** - Set explicitly if needed

**Solution:** Match old behavior:
```yaml
data:
  train_split: 1.0  # No validation (like V1)

optimization:
  warmup_steps: 0  # No warmup
  scheduler_type: "constant"  # Constant LR
```

### Issue: "Out of memory with same batch size"

**Cause:** V2 uses gradient checkpointing by default

**Solution:**
```yaml
training:
  gradient_checkpointing: false  # Match V1 behavior
```

Or keep it enabled and reduce batch size (gradient checkpointing is better!).

## Rollback Plan

If you need to rollback to V1:

1. The old files are still present:
   - `complete_optimized_training.py`
   - `ryzen_4090_optimized_training.py`

2. Simply use the old import:
```python
from complete_optimized_training import CompleteOptimizedTrainer
```

3. Both systems can coexist:
```python
# V1 training
from complete_optimized_training import CompleteOptimizedTrainer
trainer_v1 = CompleteOptimizedTrainer(...)

# V2 training
from training import ModularTrainer
trainer_v2 = ModularTrainer(...)
```

## Testing Migration

Validate your migration with a quick test run:

```bash
# Use fast iteration config for quick test
python train_v2.py --config configs/fast_iteration_config.yaml
```

This should complete in ~10 minutes and verify:
- ✅ Configuration loading
- ✅ Data loading
- ✅ Model training
- ✅ Validation
- ✅ Checkpointing
- ✅ Metrics calculation

## Support

If you encounter issues:

1. Check this migration guide
2. Review `TRAINING_V2_README.md` for detailed documentation
3. Check configuration examples in `configs/`
4. Open an issue with:
   - Your config YAML
   - Error message
   - Expected vs actual behavior

## Checklist

- [ ] Install new dependencies (`pip install -r training_requirements.txt`)
- [ ] Create config file from template
- [ ] Update training script to use new imports
- [ ] Test with fast iteration config
- [ ] Run full training
- [ ] Verify results match V1 (if needed)
- [ ] Enable new features (validation, metrics, etc.)
- [ ] Commit config to version control
- [ ] Update documentation/README

## Summary

The V2 training system provides:
- ✅ Cleaner, more maintainable code
- ✅ Type-safe configuration
- ✅ Better metrics and validation
- ✅ LoRA support for large models
- ✅ Extensible callback system
- ✅ Version-controlled experiments

Migration effort: ~30 minutes for basic setup, 1-2 hours for full adoption of new features.
