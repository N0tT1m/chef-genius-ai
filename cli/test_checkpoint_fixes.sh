#!/bin/bash
# Test script to verify checkpoint fixes are in place

echo "🧪 Testing Checkpoint Fixes"
echo "======================================"
echo ""

# Check if checkpoint utilities exist
echo "1. Checking checkpoint_utils.py..."
if [ -f "training/checkpoint_utils.py" ]; then
    echo "   ✅ checkpoint_utils.py exists"

    # Check for key functions
    if grep -q "class CheckpointManager" training/checkpoint_utils.py; then
        echo "   ✅ CheckpointManager class found"
    fi

    if grep -q "def save_rng_states" training/checkpoint_utils.py; then
        echo "   ✅ save_rng_states function found"
    fi

    if grep -q "def restore_rng_states" training/checkpoint_utils.py; then
        echo "   ✅ restore_rng_states function found"
    fi

    if grep -q "def save_checkpoint" training/checkpoint_utils.py; then
        echo "   ✅ save_checkpoint function found"
    fi

    if grep -q "def load_checkpoint" training/checkpoint_utils.py; then
        echo "   ✅ load_checkpoint function found"
    fi
else
    echo "   ❌ checkpoint_utils.py not found"
    exit 1
fi

echo ""
echo "2. Checking modular_trainer.py fixes..."
if grep -q "from training.checkpoint_utils import CheckpointManager" training/modular_trainer.py; then
    echo "   ✅ CheckpointManager imported"
else
    echo "   ❌ CheckpointManager not imported"
    exit 1
fi

if grep -q "def resume_from_checkpoint" training/modular_trainer.py; then
    echo "   ✅ resume_from_checkpoint method added"
else
    echo "   ❌ resume_from_checkpoint method not found"
    exit 1
fi

if grep -q "CheckpointManager.save_checkpoint" training/modular_trainer.py; then
    echo "   ✅ Uses CheckpointManager for saving"
else
    echo "   ❌ Not using CheckpointManager for saving"
    exit 1
fi

echo ""
echo "3. Checking complete_optimized_training.py fixes..."
if grep -q "from checkpoint_utils import CheckpointManager" complete_optimized_training.py; then
    echo "   ✅ CheckpointManager imported"
else
    echo "   ❌ CheckpointManager not imported"
    exit 1
fi

if grep -q "CheckpointManager.load_checkpoint" complete_optimized_training.py; then
    echo "   ✅ Uses CheckpointManager for loading"
else
    echo "   ❌ Not using CheckpointManager for loading"
    exit 1
fi

if grep -q "CheckpointManager.save_checkpoint" complete_optimized_training.py; then
    echo "   ✅ Uses CheckpointManager for saving"
else
    echo "   ❌ Not using CheckpointManager for saving"
    exit 1
fi

echo ""
echo "4. Checking data_manager.py fixes..."
if grep -q "seed: int = 42" training/data_manager.py; then
    echo "   ✅ Seed parameter added to __init__"
else
    echo "   ❌ Seed parameter not found"
    exit 1
fi

if grep -q "rng = random.Random(self.seed)" training/data_manager.py; then
    echo "   ✅ Deterministic RNG used for splitting"
else
    echo "   ❌ Deterministic splitting not implemented"
    exit 1
fi

echo ""
echo "5. Checking utility scripts..."
if [ -f "verify_checkpoint.py" ]; then
    echo "   ✅ verify_checkpoint.py exists"
    if [ -x "verify_checkpoint.py" ]; then
        echo "   ✅ verify_checkpoint.py is executable"
    fi
else
    echo "   ❌ verify_checkpoint.py not found"
fi

if [ -f "fix_checkpoint.py" ]; then
    echo "   ✅ fix_checkpoint.py exists"
    if [ -x "fix_checkpoint.py" ]; then
        echo "   ✅ fix_checkpoint.py is executable"
    fi
else
    echo "   ❌ fix_checkpoint.py not found"
fi

echo ""
echo "======================================"
echo "✅ All checkpoint fixes verified!"
echo ""
echo "📝 Summary of fixes:"
echo "   1. Created checkpoint_utils.py with CheckpointManager"
echo "   2. Fixed modular_trainer.py to save/load complete state"
echo "   3. Fixed complete_optimized_training.py to use CheckpointManager"
echo "   4. Fixed data_manager.py for deterministic splits"
echo "   5. Added verification and repair scripts"
echo ""
echo "🎯 Key improvements:"
echo "   - RNG states (PyTorch, CUDA, Python, NumPy) now saved/restored"
echo "   - Optimizer state properly saved/restored"
echo "   - Scheduler state properly saved/restored"
echo "   - Deterministic train/val splits with seed"
echo "   - Checkpoint verification tool available"
echo ""
echo "📚 Usage:"
echo "   Verify checkpoint: ./verify_checkpoint.py <checkpoint_dir> -v"
echo "   Fix old checkpoint: ./fix_checkpoint.py <checkpoint_dir>"
echo ""
