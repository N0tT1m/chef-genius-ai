#!/usr/bin/env python3
"""
Verify and fix checkpoint integrity by creating a basic training_state.pt file.

This script checks if a checkpoint has the required training_state.pt file,
and if not, creates a basic one to allow resuming training.
"""

import os
import sys
import torch
import argparse
from pathlib import Path


def verify_checkpoint(checkpoint_dir: str, create_if_missing: bool = False):
    """
    Verify checkpoint integrity and optionally create missing training_state.pt.

    Args:
        checkpoint_dir: Path to the checkpoint directory
        create_if_missing: If True, create a basic training_state.pt if missing

    Returns:
        bool: True if checkpoint is valid or was fixed, False otherwise
    """
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False

    print(f"📂 Checking checkpoint: {checkpoint_dir}")

    # Required files for model
    required_model_files = [
        "config.json",
        "generation_config.json",
        "model.safetensors",  # or pytorch_model.bin
        "tokenizer.json",
        "tokenizer_config.json"
    ]

    # Check model files
    missing_files = []
    for file in required_model_files:
        file_path = checkpoint_path / file
        if not file_path.exists():
            # Check alternative for model weights
            if file == "model.safetensors":
                alt_path = checkpoint_path / "pytorch_model.bin"
                if not alt_path.exists():
                    missing_files.append(file)
            else:
                missing_files.append(file)
        else:
            print(f"  ✅ {file}")

    if missing_files:
        print(f"\n❌ Missing required files:")
        for file in missing_files:
            print(f"  - {file}")
        return False

    # Check training state
    training_state_path = checkpoint_path / "training_state.pt"

    if training_state_path.exists():
        print(f"\n✅ training_state.pt found")

        # Verify it can be loaded
        try:
            state = torch.load(training_state_path, map_location='cpu')
            print(f"  📊 Training state contents:")
            print(f"    - Global step: {state.get('global_step', 'N/A')}")
            print(f"    - Epoch: {state.get('epoch', 'N/A')}")
            print(f"    - Best loss: {state.get('best_loss', 'N/A')}")
            print(f"    - Optimizer state: {'✅' if 'optimizer_state_dict' in state else '❌'}")
            print(f"    - Scheduler state: {'✅' if 'scheduler_state_dict' in state else '❌'}")
            return True
        except Exception as e:
            print(f"  ❌ Error loading training_state.pt: {e}")
            if not create_if_missing:
                return False
            print(f"  🔧 Will recreate training_state.pt...")
    else:
        print(f"\n⚠️  training_state.pt NOT found")
        if not create_if_missing:
            print(f"\n💡 Run with --fix to create a basic training_state.pt")
            return False

    # Create basic training state
    if create_if_missing:
        print(f"\n🔧 Creating basic training_state.pt...")

        # Extract step number from checkpoint directory name
        checkpoint_name = checkpoint_path.name
        if checkpoint_name.startswith("checkpoint-"):
            try:
                step = int(checkpoint_name.split("-")[1])
            except (IndexError, ValueError):
                step = 0
                print(f"  ⚠️  Could not extract step from '{checkpoint_name}', using 0")
        else:
            step = 0
            print(f"  ⚠️  Non-standard checkpoint name '{checkpoint_name}', using step 0")

        # Calculate approximate epoch (assuming 1000 steps per checkpoint, ~3000 steps per epoch)
        estimated_epoch = step // 3000

        print(f"  📊 Inferred values:")
        print(f"    - Global step: {step}")
        print(f"    - Estimated epoch: {estimated_epoch}")

        # Create minimal training state with empty (but valid) optimizer/scheduler dicts
        # This allows the training script to properly restore them
        # The optimizer and scheduler will use these empty states and effectively restart
        training_state = {
            'global_step': step,
            'epoch': estimated_epoch,
            'optimizer_state_dict': {},  # Empty dict - will cause optimizer to restart
            'scheduler_state_dict': {},  # Empty dict - will cause scheduler to restart
            'epoch_losses': [],
            'best_loss': float('inf')
        }

        # Save the training state
        try:
            torch.save(training_state, training_state_path)
            print(f"\n✅ Created basic training_state.pt")
            print(f"\n⚠️  NOTE: Optimizer and scheduler states are missing.")
            print(f"    Training will resume from step {step}, but learning rate")
            print(f"    schedule will restart. This is suboptimal but allows continuation.")
            return True
        except Exception as e:
            print(f"\n❌ Failed to create training_state.pt: {e}")
            return False

    return False


def main():
    parser = argparse.ArgumentParser(
        description='Verify and optionally fix checkpoint integrity',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Just verify checkpoint
  python verify_and_fix_checkpoint.py models/checkpoint-30000

  # Verify and fix if needed
  python verify_and_fix_checkpoint.py models/checkpoint-30000 --fix
        """
    )

    parser.add_argument(
        'checkpoint_dir',
        help='Path to checkpoint directory'
    )

    parser.add_argument(
        '--fix',
        action='store_true',
        help='Create basic training_state.pt if missing (allows resuming with fresh optimizer/scheduler)'
    )

    args = parser.parse_args()

    success = verify_checkpoint(args.checkpoint_dir, create_if_missing=args.fix)

    if success:
        print(f"\n✅ Checkpoint is valid and ready to use!")
        sys.exit(0)
    else:
        print(f"\n❌ Checkpoint validation failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
