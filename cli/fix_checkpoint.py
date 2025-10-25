#!/usr/bin/env python3
"""
Checkpoint Repair Script
Repairs old checkpoints by adding missing RNG state and proper structure.
"""

import os
import sys
import torch
import argparse
import shutil
from pathlib import Path

# Add training module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from checkpoint_utils import CheckpointManager


def fix_checkpoint(checkpoint_dir: str, backup: bool = True) -> bool:
    """
    Fix an old checkpoint by adding missing RNG states.

    Args:
        checkpoint_dir: Path to checkpoint directory
        backup: Whether to create a backup before fixing

    Returns:
        True if successful, False otherwise
    """
    print(f"🔧 Fixing checkpoint: {checkpoint_dir}\n")

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return False

    # Verify current state
    results = CheckpointManager.verify_checkpoint(checkpoint_dir)

    if not results['model_config'] or not results['model_weights']:
        print(f"❌ Checkpoint is missing model files - cannot fix")
        return False

    if not results['training_state']:
        print(f"⚠️  No training_state.pt found - creating minimal state")
        print(f"   Note: This will work but optimizer/scheduler will reset")

        # Create minimal training state
        training_state = {
            'global_step': 0,
            'epoch': 0,
            'best_loss': float('inf'),
            'epoch_losses': [],
            'optimizer_state_dict': {},
            'scheduler_state_dict': {},
            'rng_states': CheckpointManager.save_rng_states()
        }

        training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
        torch.save(training_state, training_state_path)
        print(f"✅ Created training_state.pt with current RNG states")
        return True

    # Load existing training state
    training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')

    # Create backup if requested
    if backup:
        backup_path = training_state_path + '.backup'
        shutil.copy2(training_state_path, backup_path)
        print(f"📦 Created backup: {backup_path}")

    try:
        state = torch.load(training_state_path, map_location='cpu')

        print("📊 Current checkpoint state:")
        print(f"  Global step: {state.get('global_step', 'N/A')}")
        print(f"  Epoch: {state.get('epoch', 'N/A')}")
        print(f"  Has optimizer state: {bool(state.get('optimizer_state_dict'))}")
        print(f"  Has scheduler state: {bool(state.get('scheduler_state_dict'))}")
        print(f"  Has RNG states: {bool(state.get('rng_states'))}")

        # Check what needs fixing
        needs_fix = False

        if not state.get('rng_states'):
            print("\n⚠️  Missing RNG states - will add current RNG states")
            print("   WARNING: This won't make the checkpoint deterministic for past runs,")
            print("   but will enable deterministic training from this point forward.")
            state['rng_states'] = CheckpointManager.save_rng_states()
            needs_fix = True

        # Ensure all required fields are present
        if 'epoch_losses' not in state:
            state['epoch_losses'] = []
            needs_fix = True

        if 'best_loss' not in state:
            state['best_loss'] = float('inf')
            needs_fix = True

        if needs_fix:
            # Save fixed state
            torch.save(state, training_state_path)
            print("\n✅ Checkpoint fixed successfully!")

            # Verify the fix
            print("\n🔍 Verifying fix...")
            results = CheckpointManager.verify_checkpoint(checkpoint_dir)

            if results['rng_states']:
                print("✅ RNG states now present")
            else:
                print("❌ Fix verification failed")
                return False

            return True
        else:
            print("\n✅ Checkpoint already has all required components!")
            return True

    except Exception as e:
        print(f"\n❌ Error fixing checkpoint: {e}")
        import traceback
        traceback.print_exc()

        # Restore backup if it exists
        if backup:
            backup_path = training_state_path + '.backup'
            if os.path.exists(backup_path):
                shutil.copy2(backup_path, training_state_path)
                print(f"🔄 Restored from backup")

        return False


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Fix old checkpoints by adding missing RNG states'
    )
    parser.add_argument(
        'checkpoint_dir',
        type=str,
        help='Path to checkpoint directory to fix'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip creating backup (not recommended)'
    )

    args = parser.parse_args()

    success = fix_checkpoint(
        checkpoint_dir=args.checkpoint_dir,
        backup=not args.no_backup
    )

    if success:
        print("\n✅ Checkpoint is now ready for stable resume!")
        sys.exit(0)
    else:
        print("\n❌ Failed to fix checkpoint")
        sys.exit(1)


if __name__ == "__main__":
    main()
