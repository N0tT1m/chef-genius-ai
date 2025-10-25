#!/usr/bin/env python3
"""
Checkpoint Verification Script
Verifies checkpoint integrity and displays complete state information.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Dict, Any

# Add training module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'training'))
from checkpoint_utils import CheckpointManager


def format_bytes(bytes_size: int) -> str:
    """Format bytes to human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} TB"


def verify_checkpoint(checkpoint_dir: str, verbose: bool = False) -> None:
    """
    Verify checkpoint integrity and display information.

    Args:
        checkpoint_dir: Path to checkpoint directory
        verbose: Show detailed information
    """
    print(f"🔍 Verifying checkpoint: {checkpoint_dir}\n")

    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint directory not found: {checkpoint_dir}")
        return

    # Run verification
    results = CheckpointManager.verify_checkpoint(checkpoint_dir)

    # Display results
    print("📋 Checkpoint Components:")
    print("=" * 50)

    components = [
        ("Model Config", results['model_config']),
        ("Model Weights", results['model_weights']),
        ("Tokenizer", results['tokenizer']),
        ("Training State", results['training_state']),
        ("Optimizer State", results['optimizer_state']),
        ("Scheduler State", results['scheduler_state']),
        ("RNG States", results['rng_states']),
    ]

    for name, present in components:
        status = "✅" if present else "❌"
        print(f"  {status} {name}")

    # Overall status
    print("\n" + "=" * 50)
    required_components = ['model_config', 'model_weights', 'training_state']
    optional_components = ['optimizer_state', 'scheduler_state', 'rng_states']

    all_required = all(results[comp] for comp in required_components)
    all_optional = all(results[comp] for comp in optional_components)

    if all_required and all_optional:
        print("✅ Checkpoint is COMPLETE and ready for stable resume")
    elif all_required:
        print("⚠️  Checkpoint is PARTIAL - missing some state")
        missing = [comp for comp in optional_components if not results[comp]]
        print(f"   Missing: {', '.join(missing)}")
        print("   ⚠️  Resume may cause training instability!")
    else:
        print("❌ Checkpoint is INCOMPLETE - cannot resume training")
        return

    # Load and display training state details if verbose
    if verbose and results['training_state']:
        print("\n📊 Training State Details:")
        print("=" * 50)

        training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
        try:
            state = torch.load(training_state_path, map_location='cpu')

            # Display basic info
            print(f"  Global Step: {state.get('global_step', 'N/A'):,}")
            print(f"  Epoch: {state.get('epoch', 'N/A')}")
            print(f"  Best Loss: {state.get('best_loss', float('inf')):.6f}")

            epoch_losses = state.get('epoch_losses', [])
            if epoch_losses:
                print(f"  Epoch Losses: {len(epoch_losses)} epochs")
                if len(epoch_losses) <= 10:
                    for i, loss in enumerate(epoch_losses, 1):
                        print(f"    Epoch {i}: {loss:.6f}")
                else:
                    print(f"    First: {epoch_losses[0]:.6f}")
                    print(f"    Last: {epoch_losses[-1]:.6f}")
                    print(f"    Best: {min(epoch_losses):.6f}")

            # RNG state info
            rng_states = state.get('rng_states', {})
            if rng_states:
                print(f"\n  RNG States Present:")
                for rng_type in ['python_rng_state', 'numpy_rng_state', 'torch_rng_state', 'cuda_rng_state']:
                    present = rng_type in rng_states
                    status = "✅" if present else "❌"
                    print(f"    {status} {rng_type}")

            # Optimizer state info
            optimizer_state = state.get('optimizer_state_dict', {})
            if optimizer_state:
                print(f"\n  Optimizer State:")
                print(f"    State groups: {len(optimizer_state.get('state', {}))}")
                print(f"    Param groups: {len(optimizer_state.get('param_groups', []))}")

            # Scheduler state info
            scheduler_state = state.get('scheduler_state_dict', {})
            if scheduler_state:
                print(f"\n  Scheduler State:")
                for key, value in scheduler_state.items():
                    if key == 'last_epoch':
                        print(f"    Last epoch: {value}")
                    elif key == '_step_count':
                        print(f"    Step count: {value}")
                    elif key == '_last_lr':
                        if isinstance(value, list) and len(value) > 0:
                            print(f"    Last LR: {value[0]:.2e}")

            # File size info
            file_size = os.path.getsize(training_state_path)
            print(f"\n  Training state file size: {format_bytes(file_size)}")

        except Exception as e:
            print(f"  ❌ Error loading training state: {e}")

    # File size summary
    print("\n💾 File Sizes:")
    print("=" * 50)

    total_size = 0
    for root, dirs, files in os.walk(checkpoint_dir):
        for file in files:
            file_path = os.path.join(root, file)
            file_size = os.path.getsize(file_path)
            total_size += file_size

            if verbose:
                rel_path = os.path.relpath(file_path, checkpoint_dir)
                print(f"  {rel_path}: {format_bytes(file_size)}")

    print(f"\n  Total checkpoint size: {format_bytes(total_size)}")

    print("\n" + "=" * 50)
    print("✅ Verification complete!\n")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Verify checkpoint integrity and display state information'
    )
    parser.add_argument(
        'checkpoint_dir',
        type=str,
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Show detailed information'
    )

    args = parser.parse_args()

    verify_checkpoint(args.checkpoint_dir, verbose=args.verbose)


if __name__ == "__main__":
    main()
