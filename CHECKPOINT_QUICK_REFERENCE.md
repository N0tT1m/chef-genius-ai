# Checkpoint Quick Reference Card

## The Problem
When resuming from checkpoint, training becomes unstable and loss jumps up and down.

## The Fix
All checkpoint issues have been fixed. Checkpoints now save:
- ✅ Model weights
- ✅ Optimizer state (momentum, Adam moments)
- ✅ Scheduler state (learning rate, step counts)
- ✅ RNG states (PyTorch, CUDA, Python, NumPy)
- ✅ Training progress (epoch, step, loss history)

## Quick Commands

### Verify a Checkpoint
```bash
cd cli
./verify_checkpoint.py ./output/checkpoint-1000 -v
```

### Fix an Old Checkpoint
```bash
cd cli
./fix_checkpoint.py ./output/checkpoint-1000
```

### Resume Training (modular_trainer.py)
```python
trainer.train(resume_checkpoint="./output/checkpoint-1000")
```

### Resume Training (complete_optimized_training.py)
```bash
python complete_optimized_training.py \
    --resume-from-checkpoint ./output/checkpoint-1000 \
    --model-output ./output \
    --epochs 3 \
    --batch-size 8
```

### Test All Fixes Are In Place
```bash
cd cli
./test_checkpoint_fixes.sh
```

## What to Expect

### Good Checkpoint (COMPLETE)
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

### Partial Checkpoint (Needs Fixing)
```
✅ Model Config
✅ Model Weights
✅ Tokenizer
✅ Training State
❌ Optimizer State
❌ Scheduler State
❌ RNG States

⚠️ Checkpoint is PARTIAL - missing some state
   Missing: optimizer_state, scheduler_state, rng_states
   ⚠️ Resume may cause training instability!
```

**Action:** Run `./fix_checkpoint.py <checkpoint_dir>`

## Files Changed

### New Files
- `cli/training/checkpoint_utils.py` - Core checkpoint management
- `cli/verify_checkpoint.py` - Verification tool
- `cli/fix_checkpoint.py` - Repair tool
- `cli/test_checkpoint_fixes.sh` - Test suite

### Modified Files
- `cli/training/modular_trainer.py` - Full checkpoint support
- `cli/complete_optimized_training.py` - RNG state support
- `cli/training/data_manager.py` - Deterministic splits

## Common Issues & Solutions

### Issue: "Missing RNG states" warning
**Solution:** `./fix_checkpoint.py <checkpoint_dir>`

### Issue: Loss still jumps after resume
**Check:**
1. Run `./verify_checkpoint.py <checkpoint_dir>` - Should show COMPLETE
2. Same batch size?
3. Same seed? (should see "🎲 Deterministic shuffle" in logs)

### Issue: Different train/val split
**Check:** Look for "🎲 Deterministic shuffle with seed=42" in training logs

### Issue: Verification fails
**Try:** `./fix_checkpoint.py <checkpoint_dir>`

## Best Practices

1. **Always verify before long resume:**
   ```bash
   ./verify_checkpoint.py <checkpoint> -v
   ```

2. **Fix old checkpoints before using:**
   ```bash
   ./fix_checkpoint.py <checkpoint>
   ```

3. **Test resume with short run:**
   - Train 100 steps → save → resume → check loss continues smoothly

4. **Keep checkpoint backups:**
   - The fix script creates `.backup` files automatically

## Technical Details

### What Gets Saved
```python
training_state.pt contains:
{
    'global_step': 1000,
    'epoch': 1,
    'best_loss': 1.234,
    'epoch_losses': [1.5, 1.3, 1.2],
    'optimizer_state_dict': {...},  # Momentum, Adam moments
    'scheduler_state_dict': {...},  # LR schedule
    'rng_states': {                 # All RNG states
        'python_rng_state': ...,
        'numpy_rng_state': ...,
        'torch_rng_state': ...,
        'cuda_rng_state': ...,
        'cuda_rng_state_all': [...]
    }
}
```

### Why Each Component Matters

| Component | Without It | With It |
|-----------|-----------|---------|
| **RNG States** | Non-deterministic: different dropout, shuffling, sampling | Deterministic: exact continuation |
| **Optimizer State** | Momentum resets → loss spikes | Smooth loss continuation |
| **Scheduler State** | Wrong LR → training instability | Correct LR progression |
| **Training Progress** | Restart from epoch 0 | Continue from exact step |

## One-Minute Workflow

**Before training:**
```bash
cd cli
./test_checkpoint_fixes.sh  # Verify fixes are in place
```

**During training:**
- Checkpoints auto-save every N steps
- All state automatically saved

**To resume:**
```python
# For modular_trainer.py
trainer.train(resume_checkpoint="./output/checkpoint-1000")

# For complete_optimized_training.py
--resume-from-checkpoint ./output/checkpoint-1000
```

**To verify:**
```bash
./verify_checkpoint.py ./output/checkpoint-1000 -v
```

**If old checkpoint:**
```bash
./fix_checkpoint.py ./output/checkpoint-1000
```

## Summary

| Before | After |
|--------|-------|
| ❌ Loss jumps after resume | ✅ Stable loss continuation |
| ❌ Non-deterministic | ✅ Fully deterministic |
| ❌ Optimizer resets | ✅ Optimizer state preserved |
| ❌ No RNG states | ✅ All RNG states captured |
| ❌ Random data splits | ✅ Deterministic splits |

**Result:** Training now resumes exactly where it left off with stable loss progression.

---

For detailed information, see `CHECKPOINT_FIXES_README.md`
