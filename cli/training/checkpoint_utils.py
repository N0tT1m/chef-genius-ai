#!/usr/bin/env python3
"""
Checkpoint Utilities Module
Handles complete checkpoint state management including RNG states.
"""

import os
import random
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CheckpointState:
    """Complete training state for checkpointing."""
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    scheduler_state_dict: Dict[str, Any]
    epoch: int
    global_step: int
    best_loss: float
    epoch_losses: list
    rng_states: Dict[str, Any]


class CheckpointManager:
    """
    Manages complete checkpoint state including:
    - Model weights
    - Optimizer state (momentum, adaptive learning rates)
    - Scheduler state (learning rate, step counts)
    - Training progress (epoch, global_step, losses)
    - RNG states (PyTorch, CUDA, Python, NumPy)
    """

    @staticmethod
    def save_rng_states() -> Dict[str, Any]:
        """
        Capture all random number generator states for reproducibility.

        Returns:
            Dictionary containing all RNG states
        """
        rng_states = {
            'python_rng_state': random.getstate(),
            'numpy_rng_state': np.random.get_state(),
            'torch_rng_state': torch.get_rng_state(),
        }

        # Save CUDA RNG states if CUDA is available
        if torch.cuda.is_available():
            rng_states['cuda_rng_state'] = torch.cuda.get_rng_state()
            # Save states for all GPUs
            rng_states['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()

        return rng_states

    @staticmethod
    def restore_rng_states(rng_states: Dict[str, Any]) -> None:
        """
        Restore all random number generator states.

        Args:
            rng_states: Dictionary containing RNG states
        """
        if 'python_rng_state' in rng_states:
            random.setstate(rng_states['python_rng_state'])

        if 'numpy_rng_state' in rng_states:
            np.random.set_state(rng_states['numpy_rng_state'])

        if 'torch_rng_state' in rng_states:
            torch.set_rng_state(rng_states['torch_rng_state'])

        # Restore CUDA RNG states if available
        if torch.cuda.is_available():
            if 'cuda_rng_state' in rng_states:
                torch.cuda.set_rng_state(rng_states['cuda_rng_state'])

            if 'cuda_rng_state_all' in rng_states:
                torch.cuda.set_rng_state_all(rng_states['cuda_rng_state_all'])

    @staticmethod
    def save_checkpoint(
        checkpoint_dir: str,
        model,
        optimizer,
        scheduler,
        epoch: int,
        global_step: int,
        best_loss: float = float('inf'),
        epoch_losses: list = None,
        tokenizer = None,
        additional_state: Dict[str, Any] = None
    ) -> str:
        """
        Save complete training checkpoint with all necessary state.

        Args:
            checkpoint_dir: Directory to save checkpoint
            model: Model to save
            optimizer: Optimizer with state
            scheduler: Learning rate scheduler
            epoch: Current epoch
            global_step: Global training step
            best_loss: Best loss achieved so far
            epoch_losses: List of epoch losses
            tokenizer: Optional tokenizer to save
            additional_state: Optional additional state to save

        Returns:
            Path to saved checkpoint
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Handle torch.compile() wrapped models
        model_to_save = model
        if hasattr(model, '_orig_mod'):
            print(f"  📦 Unwrapping torch.compile() model...")
            model_to_save = model._orig_mod

        # Save model weights
        try:
            model_to_save.save_pretrained(checkpoint_dir)
            print(f"  ✅ Saved model weights")
        except Exception as e:
            print(f"  ❌ Error saving model: {e}")
            raise

        # Save tokenizer if provided
        if tokenizer is not None:
            try:
                tokenizer.save_pretrained(checkpoint_dir)
                print(f"  ✅ Saved tokenizer")
            except Exception as e:
                print(f"  ⚠️  Could not save tokenizer: {e}")

        # Capture RNG states
        rng_states = CheckpointManager.save_rng_states()

        # Create training state dictionary
        training_state = {
            'global_step': global_step,
            'epoch': epoch,
            'best_loss': best_loss,
            'epoch_losses': epoch_losses if epoch_losses is not None else [],
            'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else {},
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else {},
            'rng_states': rng_states,
        }

        # Add any additional state
        if additional_state:
            training_state.update(additional_state)

        # Save training state
        training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
        try:
            torch.save(training_state, training_state_path)

            # Verify the save
            if os.path.exists(training_state_path):
                file_size = os.path.getsize(training_state_path)
                print(f"  ✅ Saved training state ({file_size:,} bytes)")
                print(f"     - Epoch: {epoch}")
                print(f"     - Global step: {global_step}")
                print(f"     - Best loss: {best_loss:.4f}")
                print(f"     - RNG states: {'✓' if rng_states else '✗'}")
                print(f"     - Optimizer state: {'✓' if optimizer else '✗'}")
                print(f"     - Scheduler state: {'✓' if scheduler else '✗'}")
            else:
                raise RuntimeError(f"Training state file not created at {training_state_path}")

        except Exception as e:
            print(f"  ❌ Error saving training state: {e}")
            raise

        print(f"💾 Complete checkpoint saved: {checkpoint_dir}")
        return checkpoint_dir

    @staticmethod
    def load_checkpoint(
        checkpoint_dir: str,
        model,
        optimizer = None,
        scheduler = None,
        device: str = 'cuda',
        strict: bool = True
    ) -> Dict[str, Any]:
        """
        Load complete training checkpoint and restore all state.

        Args:
            checkpoint_dir: Directory containing checkpoint
            model: Model to load weights into
            optimizer: Optional optimizer to restore state
            scheduler: Optional scheduler to restore state
            device: Device to load tensors to
            strict: Whether to strictly enforce state dict loading

        Returns:
            Dictionary containing loaded state information
        """
        print(f"📂 Loading checkpoint from {checkpoint_dir}")

        # Load model weights
        try:
            # Handle torch.compile() wrapped models
            model_to_load = model
            if hasattr(model, '_orig_mod'):
                print(f"  📦 Unwrapping torch.compile() model for loading...")
                model_to_load = model._orig_mod

            from transformers import AutoModelForSeq2SeqLM
            loaded_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint_dir)
            model_to_load.load_state_dict(loaded_model.state_dict(), strict=strict)
            print(f"  ✅ Loaded model weights")
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            raise

        # Load training state
        training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')

        if not os.path.exists(training_state_path):
            print(f"  ⚠️  Training state not found at {training_state_path}")
            print(f"  ⚠️  Starting from scratch with loaded model weights only")
            return {
                'global_step': 0,
                'epoch': 0,
                'best_loss': float('inf'),
                'epoch_losses': [],
                'resumed': False
            }

        try:
            training_state = torch.load(training_state_path, map_location=device)

            # Restore basic training state
            global_step = training_state.get('global_step', 0)
            epoch = training_state.get('epoch', 0)
            best_loss = training_state.get('best_loss', float('inf'))
            epoch_losses = training_state.get('epoch_losses', [])

            print(f"  ✅ Loaded training state:")
            print(f"     - Epoch: {epoch}")
            print(f"     - Global step: {global_step}")
            print(f"     - Best loss: {best_loss:.4f}")

            # Restore optimizer state
            if optimizer is not None:
                optimizer_state = training_state.get('optimizer_state_dict')
                if optimizer_state and isinstance(optimizer_state, dict) and len(optimizer_state) > 0:
                    try:
                        optimizer.load_state_dict(optimizer_state)
                        print(f"     - Optimizer state: ✓ restored")
                    except Exception as e:
                        print(f"     - Optimizer state: ✗ could not restore ({e})")
                else:
                    print(f"     - Optimizer state: ✗ not available")

            # Restore scheduler state
            if scheduler is not None:
                scheduler_state = training_state.get('scheduler_state_dict')
                if scheduler_state and isinstance(scheduler_state, dict) and len(scheduler_state) > 0:
                    try:
                        scheduler.load_state_dict(scheduler_state)
                        print(f"     - Scheduler state: ✓ restored")
                    except Exception as e:
                        print(f"     - Scheduler state: ✗ could not restore ({e})")
                else:
                    print(f"     - Scheduler state: ✗ not available")

            # Restore RNG states
            rng_states = training_state.get('rng_states')
            if rng_states:
                try:
                    CheckpointManager.restore_rng_states(rng_states)
                    print(f"     - RNG states: ✓ restored")
                    print(f"       (PyTorch, CUDA, Python, NumPy)")
                except Exception as e:
                    print(f"     - RNG states: ✗ could not restore ({e})")
            else:
                print(f"     - RNG states: ✗ not available")
                print(f"       WARNING: Training will not be deterministic!")

            return {
                'global_step': global_step,
                'epoch': epoch,
                'best_loss': best_loss,
                'epoch_losses': epoch_losses,
                'resumed': True,
                'has_rng_states': bool(rng_states),
                'has_optimizer_state': bool(optimizer_state) if optimizer else False,
                'has_scheduler_state': bool(scheduler_state) if scheduler else False,
            }

        except Exception as e:
            print(f"  ❌ Error loading training state: {e}")
            import traceback
            traceback.print_exc()
            print(f"  ⚠️  Starting from scratch with model weights only")
            return {
                'global_step': 0,
                'epoch': 0,
                'best_loss': float('inf'),
                'epoch_losses': [],
                'resumed': False
            }

    @staticmethod
    def set_seed(seed: int = 42) -> None:
        """
        Set all random seeds for reproducibility.

        Args:
            seed: Random seed value
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        # Make CUDA operations deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print(f"🎲 Set all random seeds to {seed} (deterministic mode enabled)")

    @staticmethod
    def verify_checkpoint(checkpoint_dir: str) -> Dict[str, bool]:
        """
        Verify that a checkpoint contains all necessary components.

        Args:
            checkpoint_dir: Directory containing checkpoint

        Returns:
            Dictionary with verification results
        """
        results = {
            'model_config': False,
            'model_weights': False,
            'training_state': False,
            'optimizer_state': False,
            'scheduler_state': False,
            'rng_states': False,
            'tokenizer': False
        }

        # Check for model config
        config_path = os.path.join(checkpoint_dir, 'config.json')
        results['model_config'] = os.path.exists(config_path)

        # Check for model weights
        weights_path = os.path.join(checkpoint_dir, 'pytorch_model.bin')
        safetensors_path = os.path.join(checkpoint_dir, 'model.safetensors')
        results['model_weights'] = os.path.exists(weights_path) or os.path.exists(safetensors_path)

        # Check for tokenizer
        tokenizer_path = os.path.join(checkpoint_dir, 'tokenizer_config.json')
        results['tokenizer'] = os.path.exists(tokenizer_path)

        # Check training state
        training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
        if os.path.exists(training_state_path):
            results['training_state'] = True

            try:
                state = torch.load(training_state_path, map_location='cpu')
                results['optimizer_state'] = 'optimizer_state_dict' in state and bool(state['optimizer_state_dict'])
                results['scheduler_state'] = 'scheduler_state_dict' in state and bool(state['scheduler_state_dict'])
                results['rng_states'] = 'rng_states' in state and bool(state['rng_states'])
            except Exception:
                pass

        return results


if __name__ == "__main__":
    # Test the checkpoint utilities
    print("🧪 Testing CheckpointManager...")

    # Test RNG state capture/restore
    print("\n1. Testing RNG state management...")

    # Set initial seed
    CheckpointManager.set_seed(42)

    # Generate some random numbers
    python_rand = random.random()
    numpy_rand = np.random.rand()
    torch_rand = torch.rand(1).item()

    print(f"   Initial randoms: Python={python_rand:.4f}, NumPy={numpy_rand:.4f}, PyTorch={torch_rand:.4f}")

    # Capture state
    rng_states = CheckpointManager.save_rng_states()
    print(f"   ✅ Captured RNG states")

    # Generate more random numbers (state changed)
    python_rand2 = random.random()
    numpy_rand2 = np.random.rand()
    torch_rand2 = torch.rand(1).item()

    print(f"   After state change: Python={python_rand2:.4f}, NumPy={numpy_rand2:.4f}, PyTorch={torch_rand2:.4f}")

    # Restore state
    CheckpointManager.restore_rng_states(rng_states)
    print(f"   ✅ Restored RNG states")

    # Generate random numbers again (should match original next values)
    python_rand3 = random.random()
    numpy_rand3 = np.random.rand()
    torch_rand3 = torch.rand(1).item()

    print(f"   After restore: Python={python_rand3:.4f}, NumPy={numpy_rand3:.4f}, PyTorch={torch_rand3:.4f}")

    # Verify restoration worked
    if python_rand3 == python_rand2 and numpy_rand3 == numpy_rand2 and torch_rand3 == torch_rand2:
        print(f"   ✅ RNG state restoration verified!")
    else:
        print(f"   ⚠️  RNG states may not have restored correctly")

    print("\n✅ CheckpointManager test complete!")
