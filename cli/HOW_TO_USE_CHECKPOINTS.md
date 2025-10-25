# How to Use Checkpoints (After Fixes)

## Quick Start

### Resume Training

```python
# Using modular_trainer.py
trainer.train(resume_checkpoint="./output/checkpoint-1000")
```

```bash
# Using complete_optimized_training.py
python complete_optimized_training.py \
    --resume-from-checkpoint ./output/checkpoint-1000 \
    --model-output ./output \
    --epochs 3 \
    --batch-size 8
```

That's it! Everything is automatic.

## What Happens Automatically

When you resume from checkpoint, the system automatically:

1. ✅ Loads model weights
2. ✅ Restores optimizer state (momentum, Adam moments)
3. ✅ Restores scheduler state (learning rate, step count)
4. ✅ Restores RNG states (PyTorch, CUDA, Python, NumPy)
5. ✅ Restores training progress (epoch, step, loss history)
6. ✅ Uses same train/val split (deterministic seed)

**Result:** Training continues exactly where it left off with stable loss.

## Verify Checkpoint (Optional)

Before resuming, you can verify the checkpoint is complete:

```bash
cd cli
python3 verify_checkpoint.py ./output/checkpoint-1000 -v
```

**Good checkpoint:**
```
✅ Model Config
✅ Model Weights
✅ Tokenizer
✅ Training State
✅ Optimizer State
✅ Scheduler State
✅ RNG States

✅ Checkpoint is COMPLETE and ready for stable resume
```

**Partial checkpoint (needs fixing):**
```
✅ Model Config
✅ Model Weights
✅ Tokenizer
✅ Training State
❌ Optimizer State
❌ Scheduler State
❌ RNG States

⚠️ Checkpoint is PARTIAL - missing some state
```

## Fix Old Checkpoint (If Needed)

If you have checkpoints from before these fixes:

```bash
cd cli
python3 fix_checkpoint.py ./output/checkpoint-1000
```

This adds missing RNG states and ensures all required fields are present.

## Complete Examples

### Example 1: Training from Scratch

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from training.modular_trainer import ModularTrainer
from training.config import create_default_config

# Setup
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

config = create_default_config()
config.training.num_epochs = 3
config.training.checkpoint_every_n_steps = 1000
config.output_dir = "./my_model"

# Train
trainer = ModularTrainer(model, tokenizer, config)
trainer.train()

# Checkpoints auto-saved to:
# - ./my_model/checkpoint-1000
# - ./my_model/checkpoint-2000
# - etc.
```

### Example 2: Resume After Interruption

```python
# Same setup as above, then:

# Resume from specific checkpoint
trainer.train(resume_checkpoint="./my_model/checkpoint-2000")

# Training continues from step 2000 with stable loss
```

### Example 3: Using complete_optimized_training.py

```bash
# Start training
python cli/complete_optimized_training.py \
    --pretrained-model google/flan-t5-base \
    --model-output ./my_model \
    --epochs 3 \
    --batch-size 8 \
    --checkpoint-every-n-steps 1000

# Training interrupted? Resume:
python cli/complete_optimized_training.py \
    --resume-from-checkpoint ./my_model/checkpoint-2000 \
    --model-output ./my_model \
    --epochs 3 \
    --batch-size 8
```

### Example 4: Verify Before Resume

```bash
# Check checkpoint
python3 cli/verify_checkpoint.py ./my_model/checkpoint-2000 -v

# If incomplete, fix it
python3 cli/fix_checkpoint.py ./my_model/checkpoint-2000

# Verify fix
python3 cli/verify_checkpoint.py ./my_model/checkpoint-2000 -v

# Now resume training
python cli/complete_optimized_training.py \
    --resume-from-checkpoint ./my_model/checkpoint-2000 \
    --model-output ./my_model \
    --epochs 3
```

## Understanding Checkpoint Contents

Each checkpoint directory contains:

```
checkpoint-1000/
├── config.json                  # Model configuration
├── pytorch_model.bin            # Model weights (or model.safetensors)
├── tokenizer.json               # Tokenizer
├── tokenizer_config.json        # Tokenizer config
├── generation_config.json       # Generation config
└── training_state.pt            # ⭐ Complete training state
```

### What's in training_state.pt

```python
{
    # Progress
    'global_step': 1000,
    'epoch': 1,
    'best_loss': 1.234,
    'epoch_losses': [1.5, 1.3, 1.2],

    # Optimizer (CRITICAL for stable loss)
    'optimizer_state_dict': {
        'state': {
            # Momentum, Adam moments, etc.
        }
    },

    # Scheduler (CRITICAL for correct LR)
    'scheduler_state_dict': {
        'last_epoch': 1,
        '_last_lr': [0.0001]
    },

    # RNG (CRITICAL for determinism)
    'rng_states': {
        'python_rng_state': ...,
        'numpy_rng_state': ...,
        'torch_rng_state': ...,
        'cuda_rng_state': ...
    }
}
```

## Common Questions

### Q: Do I need to do anything special to enable this?
**A:** No. If you're using the fixed code, everything is automatic.

### Q: Will my old checkpoints work?
**A:** Yes, but they may not have all state. Run `verify_checkpoint.py` to check, and `fix_checkpoint.py` if needed.

### Q: What if verification shows "PARTIAL"?
**A:** Run `fix_checkpoint.py` to add missing components. Training will be more stable after fixing.

### Q: Will loss be exactly the same after resume?
**A:** Yes, if the checkpoint has RNG states. The training will continue exactly where it left off.

### Q: What if I change batch size when resuming?
**A:** The checkpoint will still work, but the loss may jump slightly due to different gradient accumulation. Try to use the same batch size.

### Q: Do I need to set a seed manually?
**A:** No. The `DataManager` uses seed=42 by default for deterministic splits. RNG states are automatically captured in checkpoints.

## Troubleshooting

### Loss still jumps after resume

1. Verify checkpoint is COMPLETE:
   ```bash
   python3 cli/verify_checkpoint.py <checkpoint> -v
   ```

2. If PARTIAL, fix it:
   ```bash
   python3 cli/fix_checkpoint.py <checkpoint>
   ```

3. Check you're using same batch size

4. Check logs for "🎲 Deterministic shuffle with seed=42"

### Different train/val split each time

Check for this in logs:
```
🎲 Deterministic shuffle with seed=42
```

If missing, the `DataManager` wasn't initialized with a seed. This is fixed in the new code.

### "Missing RNG states" warning

Run:
```bash
python3 cli/fix_checkpoint.py <checkpoint>
```

### Checkpoint verification fails

Common causes:
- Corrupted file
- Old checkpoint format
- Missing training_state.pt

Try to fix:
```bash
python3 cli/fix_checkpoint.py <checkpoint>
```

If that fails, the checkpoint may be corrupted. Use a previous checkpoint.

## Best Practices

1. **Always verify before long resume:**
   ```bash
   python3 cli/verify_checkpoint.py <checkpoint> -v
   ```

2. **Test resume with short run first:**
   - Train 100 steps
   - Save checkpoint
   - Resume
   - Verify loss continues smoothly

3. **Keep backups of important checkpoints:**
   ```bash
   cp -r checkpoint-5000 checkpoint-5000.backup
   ```

4. **Use consistent batch size:**
   - Same batch size for training and resume
   - Avoids gradient accumulation changes

5. **Monitor logs for determinism:**
   - Look for "🎲 Deterministic shuffle"
   - Look for "✅ RNG states restored"

## Summary

**New checkpoints (after fixes):**
- ✅ Automatically save complete state
- ✅ Automatically restore on resume
- ✅ Stable loss continuation
- ✅ Fully deterministic
- ✅ No manual intervention needed

**Old checkpoints (before fixes):**
- ⚠️ May be incomplete
- ⚠️ Can be fixed with `fix_checkpoint.py`
- ⚠️ Verify with `verify_checkpoint.py`

**To use:**
```python
# Just add this parameter:
trainer.train(resume_checkpoint="./checkpoint-1000")
```

That's it! ✅
