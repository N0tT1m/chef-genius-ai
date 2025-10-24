#!/usr/bin/env python3
"""
Batch fix all checkpoints in a model directory.

This script scans a model directory for all checkpoint-* folders and
creates missing training_state.pt files for each one.
"""

import os
import sys
import torch
import argparse
from pathlib import Path


def fix_checkpoint(checkpoint_dir: Path) -> bool:
    """
    Fix a single checkpoint by creating training_state.pt.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        bool: True if successful, False otherwise
    """
    training_state_path = checkpoint_dir / "training_state.pt"

    # Check if already has training_state.pt
    if training_state_path.exists():
        try:
            state = torch.load(training_state_path, map_location='cpu')
            print(f"  ✅ Already has valid training_state.pt (step {state.get('global_step', '?')})")
            return True
        except Exception as e:
            print(f"  ⚠️  Existing training_state.pt is corrupted: {e}")
            # Will recreate below

    # Extract step number from checkpoint directory name
    checkpoint_name = checkpoint_dir.name
    if checkpoint_name.startswith("checkpoint-"):
        try:
            step = int(checkpoint_name.split("-")[1])
        except (IndexError, ValueError):
            print(f"  ❌ Could not extract step from '{checkpoint_name}'")
            return False
    else:
        print(f"  ❌ Non-standard checkpoint name '{checkpoint_name}'")
        return False

    # Calculate approximate epoch (assuming ~3000 steps per epoch for FLAN-T5-Large)
    estimated_epoch = step // 3000

    # Create minimal training state
    training_state = {
        'global_step': step,
        'epoch': estimated_epoch,
        'optimizer_state_dict': None,  # Will be initialized from scratch
        'scheduler_state_dict': None,  # Will be initialized from scratch
        'epoch_losses': [],
        'best_loss': float('inf')
    }

    # Save the training state
    try:
        torch.save(training_state, training_state_path)
        file_size = training_state_path.stat().st_size
        print(f"  ✅ Created training_state.pt (step={step}, epoch~{estimated_epoch}, size={file_size:,} bytes)")
        return True
    except Exception as e:
        print(f"  ❌ Failed to create training_state.pt: {e}")
        return False


def find_and_fix_checkpoints(model_dir: str, dry_run: bool = False):
    """
    Find and fix all checkpoints in a model directory.

    Args:
        model_dir: Path to model directory containing checkpoint-* folders
        dry_run: If True, only scan and report, don't fix

    Returns:
        tuple: (total_found, total_fixed, total_failed)
    """
    model_path = Path(model_dir)

    if not model_path.exists():
        print(f"❌ Model directory not found: {model_dir}")
        return (0, 0, 0)

    print(f"📂 Scanning model directory: {model_dir}")

    # Find all checkpoint-* directories
    checkpoint_dirs = sorted(
        [d for d in model_path.iterdir() if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[1]) if d.name.split("-")[1].isdigit() else 0
    )

    if not checkpoint_dirs:
        print(f"⚠️  No checkpoint directories found in {model_dir}")
        return (0, 0, 0)

    print(f"📊 Found {len(checkpoint_dirs)} checkpoint(s)\n")

    total_found = len(checkpoint_dirs)
    total_fixed = 0
    total_failed = 0

    for checkpoint_dir in checkpoint_dirs:
        print(f"🔍 {checkpoint_dir.name}")

        if dry_run:
            training_state_path = checkpoint_dir / "training_state.pt"
            if training_state_path.exists():
                print(f"  ✅ Has training_state.pt")
            else:
                print(f"  ⚠️  Missing training_state.pt (would be created)")
            continue

        # Fix the checkpoint
        success = fix_checkpoint(checkpoint_dir)
        if success:
            total_fixed += 1
        else:
            total_failed += 1

    print(f"\n{'='*60}")
    print(f"📊 Summary:")
    print(f"  Total checkpoints: {total_found}")

    if not dry_run:
        print(f"  ✅ Fixed: {total_fixed}")
        if total_failed > 0:
            print(f"  ❌ Failed: {total_failed}")

    return (total_found, total_fixed, total_failed)


def main():
    parser = argparse.ArgumentParser(
        description='Batch fix all checkpoints in a model directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Scan and preview what would be fixed
  python fix_all_checkpoints.py models/chef-genius-flan-t5-large-6m-recipes --dry-run

  # Fix all checkpoints
  python fix_all_checkpoints.py models/chef-genius-flan-t5-large-6m-recipes

  # Fix checkpoints in a specific directory
  python fix_all_checkpoints.py /path/to/model/directory
        """
    )

    parser.add_argument(
        'model_dir',
        help='Path to model directory containing checkpoint-* folders'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Only scan and report, do not create files'
    )

    args = parser.parse_args()

    total_found, total_fixed, total_failed = find_and_fix_checkpoints(
        args.model_dir,
        dry_run=args.dry_run
    )

    if args.dry_run:
        print(f"\n💡 Run without --dry-run to fix the checkpoints")
        sys.exit(0)
    elif total_failed > 0:
        print(f"\n⚠️  Some checkpoints could not be fixed")
        sys.exit(1)
    else:
        print(f"\n✅ All checkpoints are now ready to use!")
        print(f"\n⚠️  NOTE: Optimizer and scheduler states will be initialized from scratch when resuming.")
        print(f"    This means learning rate will restart from the beginning, which is suboptimal")
        print(f"    but allows training to continue from the model weights.")
        sys.exit(0)


if __name__ == "__main__":
    main()
