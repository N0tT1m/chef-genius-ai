#!/usr/bin/env python3
"""
Test script to verify checkpoint saving and resumption works correctly.
This script simulates training, saves a checkpoint, and resumes to verify all state is preserved.
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add training module to path
sys.path.insert(0, str(Path(__file__).parent))

from training.checkpoint_utils import CheckpointManager


def test_checkpoint_save_load():
    """Test that checkpoint save and load preserves all training state."""

    print("🧪 Testing Checkpoint Save/Load Functionality")
    print("=" * 60)

    # Create a small model for testing
    print("\n1. Loading small test model...")
    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(f"   ✅ Model loaded on {device}")

    # Create optimizer and scheduler
    print("\n2. Creating optimizer and scheduler...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.5, total_iters=100)

    # Set initial training state
    initial_epoch = 0
    initial_step = 50000
    initial_best_loss = 0.2615
    initial_epoch_losses = [0.2614, 0.2613]

    print(f"   Initial state:")
    print(f"   - Epoch: {initial_epoch}")
    print(f"   - Global step: {initial_step}")
    print(f"   - Best loss: {initial_best_loss}")
    print(f"   - Epoch losses: {initial_epoch_losses}")

    # Create checkpoint directory
    checkpoint_dir = "/tmp/test_checkpoint_resume"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Save checkpoint
    print(f"\n3. Saving checkpoint to {checkpoint_dir}...")
    CheckpointManager.save_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=initial_epoch,
        global_step=initial_step,
        best_loss=initial_best_loss,
        epoch_losses=initial_epoch_losses,
        tokenizer=tokenizer
    )

    # Verify checkpoint files exist
    print("\n4. Verifying checkpoint files...")
    verification = CheckpointManager.verify_checkpoint(checkpoint_dir)

    all_verified = True
    for component, exists in verification.items():
        status = "✅" if exists else "❌"
        print(f"   {status} {component}: {exists}")
        if not exists and component in ['model_config', 'model_weights', 'training_state']:
            all_verified = False

    if not all_verified:
        print("\n❌ Critical checkpoint components missing!")
        return False

    # Create new instances to load into
    print("\n5. Creating fresh model/optimizer/scheduler instances...")
    model_new = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model_new = model_new.to(device)
    optimizer_new = torch.optim.AdamW(model_new.parameters(), lr=1e-4)
    scheduler_new = torch.optim.lr_scheduler.LinearLR(optimizer_new, start_factor=0.5, total_iters=100)

    # Load checkpoint
    print(f"\n6. Loading checkpoint from {checkpoint_dir}...")
    loaded_state = CheckpointManager.load_checkpoint(
        checkpoint_dir=checkpoint_dir,
        model=model_new,
        optimizer=optimizer_new,
        scheduler=scheduler_new,
        device=device
    )

    # Verify loaded state
    print("\n7. Verifying loaded state matches saved state...")

    errors = []

    if loaded_state['global_step'] != initial_step:
        errors.append(f"Global step mismatch: expected {initial_step}, got {loaded_state['global_step']}")

    if loaded_state['epoch'] != initial_epoch:
        errors.append(f"Epoch mismatch: expected {initial_epoch}, got {loaded_state['epoch']}")

    if abs(loaded_state['best_loss'] - initial_best_loss) > 1e-6:
        errors.append(f"Best loss mismatch: expected {initial_best_loss}, got {loaded_state['best_loss']}")

    if loaded_state['epoch_losses'] != initial_epoch_losses:
        errors.append(f"Epoch losses mismatch: expected {initial_epoch_losses}, got {loaded_state['epoch_losses']}")

    if not loaded_state.get('has_rng_states', False):
        errors.append("RNG states were not saved/restored")

    if not loaded_state.get('has_optimizer_state', False):
        errors.append("Optimizer state was not saved/restored")

    if not loaded_state.get('has_scheduler_state', False):
        errors.append("Scheduler state was not saved/restored")

    # Print results
    if errors:
        print("\n❌ Checkpoint verification FAILED:")
        for error in errors:
            print(f"   - {error}")
        return False
    else:
        print("\n✅ All checkpoint state verified successfully!")
        print("\n📊 Verification Summary:")
        print(f"   ✅ Global step: {loaded_state['global_step']}")
        print(f"   ✅ Epoch: {loaded_state['epoch']}")
        print(f"   ✅ Best loss: {loaded_state['best_loss']:.4f}")
        print(f"   ✅ Epoch losses: {loaded_state['epoch_losses']}")
        print(f"   ✅ RNG states: {'restored' if loaded_state['has_rng_states'] else 'missing'}")
        print(f"   ✅ Optimizer state: {'restored' if loaded_state['has_optimizer_state'] else 'missing'}")
        print(f"   ✅ Scheduler state: {'restored' if loaded_state['has_scheduler_state'] else 'missing'}")
        return True


def test_fractional_epoch_calculation():
    """Test that fractional epoch is calculated correctly."""

    print("\n\n🧪 Testing Fractional Epoch Calculation")
    print("=" * 60)

    # Simulate training scenario
    total_batches = 100000  # 100k batches per epoch
    gradient_accumulation_steps = 4

    steps_per_epoch = total_batches / gradient_accumulation_steps  # 25000 steps per epoch

    # Test case 1: At step 50000 (should be epoch 2.0)
    global_step = 50000
    fractional_epoch = global_step / steps_per_epoch
    print(f"\n1. Global step {global_step}:")
    print(f"   Fractional epoch: {fractional_epoch:.4f}")
    print(f"   Expected: 2.0000")

    if abs(fractional_epoch - 2.0) < 0.001:
        print("   ✅ Correct!")
    else:
        print("   ❌ Incorrect!")
        return False

    # Test case 2: At step 12500 (should be epoch 0.5)
    global_step = 12500
    fractional_epoch = global_step / steps_per_epoch
    print(f"\n2. Global step {global_step}:")
    print(f"   Fractional epoch: {fractional_epoch:.4f}")
    print(f"   Expected: 0.5000")

    if abs(fractional_epoch - 0.5) < 0.001:
        print("   ✅ Correct!")
    else:
        print("   ❌ Incorrect!")
        return False

    # Test case 3: At step 37500 (should be epoch 1.5)
    global_step = 37500
    fractional_epoch = global_step / steps_per_epoch
    print(f"\n3. Global step {global_step}:")
    print(f"   Fractional epoch: {fractional_epoch:.4f}")
    print(f"   Expected: 1.5000")

    if abs(fractional_epoch - 1.5) < 0.001:
        print("   ✅ Correct!")
    else:
        print("   ❌ Incorrect!")
        return False

    print("\n✅ All fractional epoch calculations correct!")
    return True


def main():
    """Run all checkpoint tests."""

    print("🚀 Checkpoint Resume Testing Suite")
    print("=" * 60)

    # Test 1: Checkpoint save/load
    test1_passed = test_checkpoint_save_load()

    # Test 2: Fractional epoch calculation
    test2_passed = test_fractional_epoch_calculation()

    # Summary
    print("\n\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Checkpoint Save/Load: {'✅ PASSED' if test1_passed else '❌ FAILED'}")
    print(f"Fractional Epoch Calc: {'✅ PASSED' if test2_passed else '❌ FAILED'}")

    if test1_passed and test2_passed:
        print("\n🎉 All tests PASSED! Checkpoint resume should work correctly.")
        return 0
    else:
        print("\n❌ Some tests FAILED. Please review the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
