# Checkpoint Saving/Loading and Training Loop Analysis
## Chef Genius Recipe Generation Training System

---

## Executive Summary

The codebase has a **modular training system** with checkpoint support split across two main implementations:

1. **Modular Trainer** (`cli/training/modular_trainer.py`) - NEW, cleaner implementation
2. **Complete Optimized Trainer** (`cli/complete_optimized_training.py`) - Older, monolithic approach

### Key Findings:

**CRITICAL ISSUES IDENTIFIED:**
1. ⚠️ **Incomplete Checkpoint State Saving** - Missing random number generator (RNG) state
2. ⚠️ **Loose Optimizer/Scheduler Validation** - Graceful fallback but can cause training instability
3. ⚠️ **torch.compile() Wrapping Issue** - Model unwrapping handled but fragile
4. ✅ **Data Shuffling** - Properly handled with fresh shuffle on resume
5. ⚠️ **Learning Rate Restoration** - Not persisted, resets on resume

---

## 1. CHECKPOINT SAVING IMPLEMENTATION

### 1.1 Modular Trainer (RECOMMENDED - `cli/training/modular_trainer.py`)

**Location:** `_save_checkpoint()` method (lines 433-448)

```python
def _save_checkpoint(self, step: int) -> None:
    """Save model checkpoint."""
    checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    if self.config.training.use_lora:
        # Save LoRA adapters
        self.model.save_pretrained(checkpoint_dir)
    else:
        # Save full model
        self.model.save_pretrained(checkpoint_dir)

    self.tokenizer.save_pretrained(checkpoint_dir)

    print(f"   💾 Checkpoint saved: {checkpoint_dir}")
```

**CRITICAL ISSUE:** This implementation **does NOT save optimizer, scheduler, or random state!**

Missing from checkpoint:
- ❌ `optimizer.state_dict()` - Adam momentum, velocity buffers
- ❌ `scheduler.state_dict()` - Learning rate schedule state
- ❌ `global_step` counter
- ❌ `torch.get_rng_state()` / `torch.cuda.get_rng_state()` - For reproducibility
- ❌ `random.getstate()` - Python RNG state
- ❌ `numpy.random.get_state()` - NumPy RNG state
- ❌ Epoch number
- ❌ Best validation loss

---

### 1.2 Complete Optimized Trainer (`cli/complete_optimized_training.py`)

**Location:** `save_checkpoint()` method (lines 1243-1296)

```python
def save_checkpoint(self, step, optimizer=None, scheduler=None, epoch=None):
    """Save training checkpoint with full training state."""
    checkpoint_dir = f"{self.output_dir}/checkpoint-{step}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save model and tokenizer
    try:
        # Handle torch.compile() wrapped models - need to access ._orig_mod
        model_to_save = self.model
        if hasattr(self.model, '_orig_mod'):
            print(f"  📦 Unwrapping torch.compile() model...")
            model_to_save = self.model._orig_mod

        model_to_save.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"💾 Saved model and tokenizer to {checkpoint_dir}")
    except Exception as e:
        print(f"❌ Error saving model/tokenizer: {e}")
        traceback.print_exc()
        raise

    # Save training state (optimizer, scheduler, step count, epoch)
    if optimizer is not None and scheduler is not None:
        try:
            training_state = {
                'global_step': step,
                'epoch': epoch if epoch is not None else self.current_epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch_losses': self.epoch_losses,
                'best_loss': getattr(self, 'best_loss', float('inf'))
            }

            training_state_path = f"{checkpoint_dir}/training_state.pt"
            print(f"  💾 Saving training state to {training_state_path}...")
            torch.save(training_state, training_state_path)

            # Verify the file was actually written
            if os.path.exists(training_state_path):
                file_size = os.path.getsize(training_state_path)
                print(f"✅ Saved training state: step {step}, epoch {epoch} ({file_size:,} bytes)")
            else:
                print(f"❌ WARNING: training_state.pt was not created at {training_state_path}")
                raise RuntimeError(f"Failed to create training_state.pt")

        except Exception as e:
            print(f"❌ Error saving training state: {e}")
            traceback.print_exc()
            # Don't raise - allow training to continue even if state save fails
            print(f"⚠️  Training will continue, but resume may not work properly")
    else:
        print(f"⚠️  Optimizer or scheduler is None - skipping training state save")

    return checkpoint_dir
```

**SAVED STATE CONTENTS:**
✅ Global step count
✅ Epoch number  
✅ Optimizer state dict
✅ Scheduler state dict
✅ Epoch losses
✅ Best loss

**MISSING STATE:**
❌ Random number generator states (torch, random, numpy)
❌ Data shuffle seed
❌ Mixed precision scaler state (if using FP16)

---

## 2. CHECKPOINT LOADING IMPLEMENTATION

### 2.1 Modular Trainer (NO RESUME SUPPORT)

**Location:** `train()` method

The modular trainer **does NOT have checkpoint resumption implemented**. It starts fresh each time.

```python
def train(self) -> None:
    """Run training loop."""
    # ... setup code ...
    
    # Create optimizer and scheduler (fresh, not from checkpoint)
    self.optimizer = self._create_optimizer()
    self.scheduler = self._create_scheduler(total_steps)
    
    # ... training loop ...
```

**ISSUE:** If you resume training with this trainer:
- ❌ Optimizer state is fresh (Adam momentum lost)
- ❌ Scheduler state is fresh (warmup restarts)
- ❌ Global step counter not restored
- ⚠️ Model weights loaded but training state completely reset

---

### 2.2 Complete Optimized Trainer (`cli/complete_optimized_training.py`)

**Location:** `train_complete_optimized()` method (lines 830-883)

```python
# Resume from checkpoint if provided
starting_step = 0
starting_epoch = 0
if resume_checkpoint:
    training_state_path = f"{resume_checkpoint}/training_state.pt"
    if os.path.exists(training_state_path):
        print(f"📂 Loading training state from {training_state_path}")
        try:
            training_state = torch.load(training_state_path, map_location='cpu')

            # Restore step and epoch counters first
            starting_step = training_state.get('global_step', 0)
            starting_epoch = training_state.get('epoch', 0)
            self.epoch_losses = training_state.get('epoch_losses', [])
            best_loss = training_state.get('best_loss', float('inf'))

            # Restore optimizer and scheduler state if they exist and are not empty
            optimizer_state = training_state.get('optimizer_state_dict')
            scheduler_state = training_state.get('scheduler_state_dict')

            if optimizer_state and isinstance(optimizer_state, dict) and len(optimizer_state) > 0:
                try:
                    optimizer.load_state_dict(optimizer_state)
                    print(f"  ✅ Restored optimizer state")
                except Exception as e:
                    print(f"  ⚠️  Could not restore optimizer state: {e}")
                    print(f"     Optimizer will restart from scratch")
            else:
                print(f"  ⚠️  Optimizer state is empty or missing - starting fresh")

            if scheduler_state and isinstance(scheduler_state, dict) and len(scheduler_state) > 0:
                try:
                    scheduler.load_state_dict(scheduler_state)
                    print(f"  ✅ Restored scheduler state")
                except Exception as e:
                    print(f"  ⚠️  Could not restore scheduler state: {e}")
                    print(f"     Scheduler will restart from scratch")
            else:
                print(f"  ⚠️  Scheduler state is empty or missing - starting fresh")

            print(f"✅ Resumed from step {starting_step}, epoch {starting_epoch}")
            if best_loss != float('inf'):
                print(f"   Previous best loss: {best_loss:.4f}")

        except Exception as e:
            print(f"❌ Error loading training state: {e}")
            print(f"   Starting from scratch with model weights from checkpoint")
            traceback.print_exc()
            best_loss = float('inf')
    else:
        print(f"⚠️  Training state not found at {training_state_path}, starting from scratch")
        best_loss = float('inf')
else:
    best_loss = float('inf')
```

**RESTORATION LOGIC:**
✅ Global step count restored
✅ Epoch number restored
✅ Optimizer state restored (with error handling)
✅ Scheduler state restored (with error handling)
✅ Best loss restored
✅ Epoch losses history restored

⚠️ **Graceful Fallback:** If optimizer/scheduler states are missing or empty, training continues with fresh states.

**MISSING:**
❌ Random number generator state restoration
❌ Data shuffle seed restoration
❌ No warning about potential instability from missing RNG states

---

## 3. TRAINING LOOP IMPLEMENTATION

### 3.1 Main Training Loop (`complete_optimized_training.py`)

**Location:** Lines 913-1053

```python
# Training loop
self.model.train()
total_steps = starting_step  # Resume from checkpoint step

try:
    for epoch in range(starting_epoch, epochs):
        self.current_epoch = epoch + 1
        
        # Reset Rust dataloader for new epoch (required for multi-epoch training)
        if hasattr(train_loader, 'reset'):
            train_loader.reset()
            print(f"🔄 Reset dataloader for epoch {self.current_epoch}")
        
        # Epoch started
        print(f"🚀 Starting epoch {self.current_epoch}/{epochs}")
        epoch_loss = 0
        batch_count = 0
        epoch_start_time = time.time()
        optimizer.zero_grad()
        
        try:
            for batch_idx, batch in enumerate(train_loader):
                # AGGRESSIVE memory management
                if batch_count % 5 == 0:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                # Move to GPU with non-blocking transfer
                batch = {k: v.to(self.model.device, non_blocking=True) if hasattr(v, 'to') else v
                        for k, v in batch.items()}
                
                # Forward pass with mixed precision
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.enable_mixed_precision):
                    model_inputs = {k: v for k, v in batch.items() if k != 'quality_scores'}
                    outputs = self.model(**model_inputs)
                    loss = outputs.loss / self.hw_config.gradient_accumulation_steps

                # Backward pass
                loss.backward()
                
                # Gradient accumulation
                batch_count += 1
                if batch_count % self.hw_config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    total_steps += 1
                    
                    # ... logging and checkpoint saving ...
```

**KEY FEATURES:**
✅ Gradient accumulation support
✅ Mixed precision training (bfloat16)
✅ Gradient clipping
✅ Learning rate scheduling with warmup
✅ Epoch and step tracking
✅ GPU memory management

---

## 4. OPTIMIZER, SCHEDULER, AND RANDOM STATE HANDLING

### 4.1 Optimizer Setup

**Type:** AdamW (fused variant available)

```python
optimizer = torch.optim.AdamW(
    self.model.parameters(),
    lr=learning_rate,
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

**State Saved:** ✅ In `training_state.pt` (Complete Optimized Trainer only)

**State Restored:** 
- ✅ With error handling in Complete Optimized Trainer
- ❌ Not supported in Modular Trainer

---

### 4.2 Learning Rate Scheduler

**Type:** Linear decay with warmup

```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps
)
```

**State Saved:** ✅ In `training_state.pt` (Complete Optimized Trainer only)

**State Restored:**
- ✅ With error handling in Complete Optimized Trainer
- ❌ Not supported in Modular Trainer

**ISSUE:** Scheduler state includes current step counter. If not restored:
- ⚠️ Learning rate resets to initial value
- ⚠️ Warmup phase restarts
- ⚠️ Linear decay restarts from beginning

---

### 4.3 Random Number Generator State

**NOT SAVED OR RESTORED** in either implementation!

**Implications:**
- ❌ Data shuffling may not be reproducible
- ❌ Model behavior may differ from original training
- ⚠️ Any dropout/stochastic components will behave differently
- ❌ Cannot reproduce exact training run

**Missing Code:**
```python
# Should save:
training_state = {
    # ... existing fields ...
    'torch_rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    'python_rng_state': random.getstate(),
    'numpy_rng_state': np.random.get_state(),
    'random_seed': self.random_seed,  # Track the seed used
}

# Should restore:
if 'torch_rng_state' in training_state:
    torch.set_rng_state(training_state['torch_rng_state'])
if 'cuda_rng_state' in training_state and torch.cuda.is_available():
    torch.cuda.set_rng_state(training_state['cuda_rng_state'])
if 'python_rng_state' in training_state:
    random.setstate(training_state['python_rng_state'])
if 'numpy_rng_state' in training_state:
    np.random.set_state(training_state['numpy_rng_state'])
```

---

## 5. DATA SHUFFLING AND SAMPLING ISSUES

### 5.1 Train/Val Split (`cli/training/data_manager.py`)

**Location:** `_create_splits()` method (lines 264-291)

```python
def _create_splits(self) -> None:
    """Create train/val splits from full dataset."""
    if self.full_dataset is None:
        raise ValueError("Dataset not loaded. Call load_datasets() first.")

    total_size = len(self.full_dataset)
    train_size = int(total_size * self.config.train_split)
    val_size = total_size - train_size

    # Create indices
    indices = list(range(total_size))
    if self.config.shuffle:
        random.shuffle(indices)  # <-- Uses Python's random module

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    # Create subsets
    self.train_dataset = Subset(self.full_dataset, train_indices)
    self.val_dataset = Subset(self.full_dataset, val_indices)
```

**ISSUE:** Uses `random.shuffle()` without seed tracking
- ⚠️ Split is non-deterministic across runs
- ❌ Different samples in train/val on each run
- ❌ Cannot reproduce exact train/val split

---

### 5.2 Batch Shuffling (DataLoader)

**Location:** `create_train_dataloader()` method (lines 312-325)

```python
def create_train_dataloader(self, batch_size: int) -> DataLoader:
    """Create training dataloader."""
    if self.train_dataset is None:
        raise ValueError("Train dataset not created. Call load_datasets() first.")

    return DataLoader(
        self.train_dataset,
        batch_size=batch_size,
        shuffle=True,  # <-- Fresh shuffle each epoch
        num_workers=0,
        collate_fn=self._collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False
    )
```

**Behavior on Resume:**
✅ Fresh shuffle applied to training data
✅ Ensures variation across resume attempts
⚠️ But changes training order from original run

---

### 5.3 Multi-Epoch Handling

**Location:** `train_complete_optimized()` method (lines 920-924)

```python
for epoch in range(starting_epoch, epochs):
    self.current_epoch = epoch + 1
    
    # Reset Rust dataloader for new epoch (required for multi-epoch training)
    if hasattr(train_loader, 'reset'):
        train_loader.reset()
        print(f"🔄 Reset dataloader for epoch {self.current_epoch}")
```

✅ Dataloader properly reset for new epochs
✅ Fresh shuffling applied

---

## ISSUES FOUND: COMMON CAUSES OF TRAINING INSTABILITY

### Issue #1: CRITICAL - Missing RNG State Persistence

**Severity:** HIGH  
**Impact:** Loss of reproducibility, non-deterministic training behavior

When resuming from checkpoint:
- New random seeds are used
- Dropout layers behave differently
- Data augmentation (if any) follows different patterns
- Cannot reproduce exact training run

**Fix Required:**
```python
# In checkpoint save
'torch_rng_state': torch.get_rng_state(),
'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
'python_rng_state': random.getstate(),

# In checkpoint load
torch.set_rng_state(state['torch_rng_state'])
if torch.cuda.is_available() and 'cuda_rng_state' in state:
    torch.cuda.set_rng_state(state['cuda_rng_state'])
random.setstate(state['python_rng_state'])
```

---

### Issue #2: Optimizer State Restoration Without Validation

**Severity:** MEDIUM  
**Impact:** Training may continue with fresh optimizer momentum

Current code gracefully handles missing optimizer state:
```python
if optimizer_state and isinstance(optimizer_state, dict) and len(optimizer_state) > 0:
    try:
        optimizer.load_state_dict(optimizer_state)
        print(f"  ✅ Restored optimizer state")
    except Exception as e:
        print(f"  ⚠️  Could not restore optimizer state: {e}")
        # Training continues with fresh optimizer!
```

**Problem:** When optimizer state is missing or corrupted:
- Adam momentum buffers are fresh (m = 0, v = 0)
- Learning effectively restarts
- Gradient updates follow fresh Adam trajectory
- Can cause loss spikes or plateaus

---

### Issue #3: Learning Rate Schedule Reset

**Severity:** MEDIUM  
**Impact:** Incorrect learning rate and warmup behavior

When scheduler state is not restored:
- Current step counter in scheduler is reset
- Learning rate may jump to warmup phase value
- Linear decay schedule restarts from beginning

**Current behavior:** Scheduler state IS saved/restored in Complete Optimized Trainer
**Modular Trainer:** Does NOT support checkpoint resume

---

### Issue #4: Inconsistent Data Splits Across Resumptions

**Severity:** LOW-MEDIUM  
**Impact:** Different training data distribution

Current code:
```python
def _create_splits(self) -> None:
    indices = list(range(total_size))
    if self.config.shuffle:
        random.shuffle(indices)  # Non-deterministic!
```

**Problem:** 
- Train/val split changes each run
- Can cause distribution shift in validation metrics
- Makes it hard to compare training runs

**Fix Required:** Use seeded random for deterministic splits
```python
indices = list(range(total_size))
if self.config.shuffle:
    rng = random.Random(self.seed)  # Use fixed seed
    rng.shuffle(indices)
```

---

### Issue #5: Modular Trainer Lacks Resume Support

**Severity:** HIGH  
**Impact:** Complete loss of checkpoint state on resume

The ModularTrainer does NOT implement checkpoint resumption:
- No global_step restoration
- No optimizer state loading
- No scheduler state loading
- Can only load model weights from previous checkpoint

**Recommended Action:** Implement checkpoint loading in ModularTrainer or use CompleteOptimizedTrainer for production

---

## 6. SPECIFIC IMPLEMENTATION ISSUES

### Complete Optimized Trainer - torch.compile() Handling

**Location:** Lines 1250-1254

```python
# Handle torch.compile() wrapped models - need to access ._orig_mod
model_to_save = self.model
if hasattr(self.model, '_orig_mod'):
    print(f"  📦 Unwrapping torch.compile() model...")
    model_to_save = self.model._orig_mod
```

✅ Properly handles torch.compile() wrapping
⚠️ But accessing `._orig_mod` is fragile (private API)

Better approach:
```python
from torch._dynamo import unwrap
model_to_save = unwrap(self.model)
```

---

### Checkpoint Verification

**Location:** Lines 1280-1286

```python
# Verify the file was actually written
if os.path.exists(training_state_path):
    file_size = os.path.getsize(training_state_path)
    print(f"✅ Saved training state: step {step}, epoch {epoch} ({file_size:,} bytes)")
else:
    print(f"❌ WARNING: training_state.pt was not created at {training_state_path}")
    raise RuntimeError(f"Failed to create training_state.pt")
```

✅ Verifies checkpoint was written to disk
✅ Catches silent save failures

---

## 7. RECOMMENDED FIXES

### Priority 1: CRITICAL - Add RNG State Persistence

```python
# Complete Optimized Trainer - update save_checkpoint()
training_state = {
    'global_step': step,
    'epoch': epoch if epoch is not None else self.current_epoch,
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch_losses': self.epoch_losses,
    'best_loss': getattr(self, 'best_loss', float('inf')),
    # NEW: RNG states
    'torch_rng_state': torch.get_rng_state(),
    'cuda_rng_state': torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    'python_rng_state': random.getstate(),
    'numpy_rng_state': np.random.get_state(),
}

# Complete Optimized Trainer - update load (lines 830-883)
if 'torch_rng_state' in training_state:
    torch.set_rng_state(training_state['torch_rng_state'])
    print(f"  ✅ Restored torch RNG state")

if 'cuda_rng_state' in training_state and torch.cuda.is_available():
    torch.cuda.set_rng_state(training_state['cuda_rng_state'])
    print(f"  ✅ Restored CUDA RNG state")

if 'python_rng_state' in training_state:
    random.setstate(training_state['python_rng_state'])
    print(f"  ✅ Restored Python RNG state")

if 'numpy_rng_state' in training_state:
    np.random.set_state(training_state['numpy_rng_state'])
    print(f"  ✅ Restored NumPy RNG state")
```

---

### Priority 2: Implement Resume in Modular Trainer

```python
# cli/training/modular_trainer.py - update __init__
def __init__(self, ...):
    # ... existing code ...
    self.checkpoint_path = config.resume_from_checkpoint
    self.starting_step = 0
    self.starting_epoch = 0

# Add new method:
def _load_checkpoint(self, checkpoint_path: str) -> None:
    """Load checkpoint state."""
    training_state_path = Path(checkpoint_path) / "training_state.pt"
    
    if training_state_path.exists():
        print(f"📂 Loading training state from {training_state_path}")
        try:
            state = torch.load(training_state_path, map_location='cpu')
            
            self.starting_step = state.get('global_step', 0)
            self.starting_epoch = state.get('epoch', 0)
            
            if self.optimizer and state.get('optimizer_state_dict'):
                self.optimizer.load_state_dict(state['optimizer_state_dict'])
                print(f"  ✅ Restored optimizer state")
            
            if self.scheduler and state.get('scheduler_state_dict'):
                self.scheduler.load_state_dict(state['scheduler_state_dict'])
                print(f"  ✅ Restored scheduler state")
            
            # Restore RNG states
            if 'torch_rng_state' in state:
                torch.set_rng_state(state['torch_rng_state'])
            
            if 'cuda_rng_state' in state and torch.cuda.is_available():
                torch.cuda.set_rng_state(state['cuda_rng_state'])
            
            print(f"✅ Resumed from step {self.starting_step}, epoch {self.starting_epoch}")
        except Exception as e:
            print(f"⚠️  Error loading checkpoint: {e}")

# Update train() loop:
def train(self) -> None:
    # ... setup ...
    
    # Load checkpoint if specified
    if self.checkpoint_path:
        self._load_checkpoint(self.checkpoint_path)
    
    # Training loop with proper resume
    for epoch in range(self.starting_epoch, self.config.training.num_epochs):
        self._train_epoch(epoch + 1, train_loader)
        # ...
```

---

### Priority 3: Use Deterministic Data Splits

```python
# cli/training/data_manager.py - DataManager.__init__
def __init__(self, tokenizer, config, seed=42):
    self.seed = seed
    # ...

# Update _create_splits()
def _create_splits(self) -> None:
    total_size = len(self.full_dataset)
    train_size = int(total_size * self.config.train_split)
    val_size = total_size - train_size

    indices = list(range(total_size))
    if self.config.shuffle:
        # Use seeded random for determinism
        rng = random.Random(self.seed)
        rng.shuffle(indices)
    
    # ... rest of code ...
```

---

## SUMMARY TABLE

| Component | Implementation | Saved | Loaded | Issue |
|-----------|---|---|---|---|
| Model Weights | Both | ✅ | ✅ | torch.compile() unwrapping needed |
| Global Step | Complete Opt. | ✅ | ✅ | None |
| Epoch Number | Complete Opt. | ✅ | ✅ | None |
| Optimizer State | Complete Opt. | ✅ | ⚠️ Graceful fallback | Can cause training instability |
| Scheduler State | Complete Opt. | ✅ | ⚠️ Graceful fallback | Learning rate may reset |
| Torch RNG | BOTH | ❌ | ❌ | CRITICAL - breaks reproducibility |
| CUDA RNG | BOTH | ❌ | ❌ | CRITICAL - breaks reproducibility |
| Python RNG | BOTH | ❌ | ❌ | CRITICAL - breaks reproducibility |
| Data Shuffle | DataManager | N/A | ⚠️ Fresh shuffle | Different train order than original |
| Train/Val Split | DataManager | ❌ | ❌ | Can differ across runs |

---

## CONCLUSION

The Chef Genius training system has:

✅ **Strengths:**
- Well-structured modular components
- Comprehensive checkpoint save/load for model and optimizer
- Graceful error handling for missing states
- torch.compile() wrapping handled
- Good logging and verification

⚠️ **Critical Issues:**
1. **Missing RNG state persistence** - Cannot reproduce training runs
2. **Modular Trainer lacks resume support** - Can only load model weights
3. **Data split non-determinism** - Train/val split differs each run
4. **Optimizer/scheduler validation** - Gracefully falls back but may cause instability

**Recommended Action:** Implement all Priority 1-3 fixes above before production use, especially RNG state persistence for reproducibility.

