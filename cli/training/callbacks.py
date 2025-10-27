#!/usr/bin/env python3
"""
Training Callbacks System
Extensible callback system for training events and monitoring.
"""

from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod
from dataclasses import dataclass
import time


@dataclass
class TrainingState:
    """Current training state passed to callbacks."""
    epoch: int
    global_step: int
    batch_idx: int
    total_batches: int
    loss: float
    learning_rate: float
    metrics: Optional[Dict[str, float]] = None
    fractional_epoch: Optional[float] = None  # Accurate epoch with fractional part


class Callback(ABC):
    """Base class for all callbacks."""

    @abstractmethod
    def on_train_begin(self, **kwargs) -> None:
        """Called at the beginning of training."""
        pass

    @abstractmethod
    def on_train_end(self, **kwargs) -> None:
        """Called at the end of training."""
        pass

    @abstractmethod
    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Called at the beginning of each epoch."""
        pass

    @abstractmethod
    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """Called at the end of each epoch."""
        pass

    @abstractmethod
    def on_batch_begin(self, batch_idx: int, **kwargs) -> None:
        """Called at the beginning of each batch."""
        pass

    @abstractmethod
    def on_batch_end(self, state: TrainingState, **kwargs) -> None:
        """Called at the end of each batch."""
        pass

    @abstractmethod
    def on_validation_begin(self, **kwargs) -> None:
        """Called at the beginning of validation."""
        pass

    @abstractmethod
    def on_validation_end(self, metrics: Dict[str, float], **kwargs) -> None:
        """Called at the end of validation."""
        pass


class BaseCallback(Callback):
    """Base callback with no-op implementations."""

    def on_train_begin(self, **kwargs) -> None:
        pass

    def on_train_end(self, **kwargs) -> None:
        pass

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        pass

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        pass

    def on_batch_begin(self, batch_idx: int, **kwargs) -> None:
        pass

    def on_batch_end(self, state: TrainingState, **kwargs) -> None:
        pass

    def on_validation_begin(self, **kwargs) -> None:
        pass

    def on_validation_end(self, metrics: Dict[str, float], **kwargs) -> None:
        pass


class ProgressCallback(BaseCallback):
    """Callback for logging training progress."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = log_every_n_steps
        self.epoch_start_time: Optional[float] = None
        self.batch_times: List[float] = []

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        self.epoch_start_time = time.time()
        self.batch_times = []
        print(f"\n🚀 Starting Epoch {epoch}")

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        if self.epoch_start_time:
            elapsed = time.time() - self.epoch_start_time
            avg_batch_time = sum(self.batch_times) / len(self.batch_times) if self.batch_times else 0
            print(f"✅ Epoch {epoch} completed in {elapsed:.1f}s (avg {avg_batch_time*1000:.1f}ms/batch)")

    def on_batch_end(self, state: TrainingState, **kwargs) -> None:
        # Track batch time
        if self.epoch_start_time:
            self.batch_times.append(time.time() - self.epoch_start_time)

        # Log progress
        if state.global_step % self.log_every_n_steps == 0:
            progress = (state.batch_idx + 1) / state.total_batches * 100
            print(f"Step {state.global_step:,} | Epoch {state.epoch} ({progress:.1f}%) | "
                  f"Loss: {state.loss:.4f} | LR: {state.learning_rate:.2e}")


class MetricsCallback(BaseCallback):
    """Callback for tracking and logging metrics."""

    def __init__(self):
        self.train_losses: List[float] = []
        self.val_metrics: List[Dict[str, float]] = []
        self.best_val_loss: float = float('inf')
        self.best_epoch: int = 0

    def on_batch_end(self, state: TrainingState, **kwargs) -> None:
        self.train_losses.append(state.loss)

    def on_validation_end(self, metrics: Dict[str, float], **kwargs) -> None:
        self.val_metrics.append(metrics)

        # Track best validation loss
        if 'val_loss' in metrics:
            val_loss = metrics['val_loss']
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = len(self.val_metrics) - 1
                print(f"✨ New best validation loss: {val_loss:.4f}")

    def get_average_train_loss(self, last_n: Optional[int] = None) -> float:
        """Get average training loss over last N steps."""
        losses = self.train_losses[-last_n:] if last_n else self.train_losses
        return sum(losses) / len(losses) if losses else 0.0


class EarlyStoppingCallback(BaseCallback):
    """Callback for early stopping based on validation metrics."""

    def __init__(
        self,
        patience: int = 3,
        min_delta: float = 0.0001,
        monitor: str = 'val_loss',
        mode: str = 'min'
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.monitor = monitor
        self.mode = mode

        self.best_score: Optional[float] = None
        self.counter = 0
        self.should_stop = False

    def on_validation_end(self, metrics: Dict[str, float], **kwargs) -> None:
        if self.monitor not in metrics:
            print(f"⚠️  Early stopping monitor metric '{self.monitor}' not found in metrics")
            return

        current_score = metrics[self.monitor]

        # Initialize best score
        if self.best_score is None:
            self.best_score = current_score
            return

        # Check for improvement
        if self.mode == 'min':
            improved = current_score < (self.best_score - self.min_delta)
        else:  # mode == 'max'
            improved = current_score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            print(f"⚠️  No improvement for {self.counter}/{self.patience} validations")

            if self.counter >= self.patience:
                print(f"🛑 Early stopping triggered! Best {self.monitor}: {self.best_score:.4f}")
                self.should_stop = True


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints."""

    def __init__(
        self,
        save_every_n_steps: int = 1000,
        keep_best_n: int = 3,
        output_dir: str = "./checkpoints"
    ):
        self.save_every_n_steps = save_every_n_steps
        self.keep_best_n = keep_best_n
        self.output_dir = output_dir
        self.checkpoints: List[tuple[str, float]] = []  # (path, score)

    def on_batch_end(self, state: TrainingState, **kwargs) -> None:
        if state.global_step % self.save_every_n_steps == 0 and state.global_step > 0:
            checkpoint_path = f"{self.output_dir}/checkpoint-{state.global_step}"
            print(f"💾 Saving checkpoint: {checkpoint_path}")
            # Actual saving is handled by trainer
            # This callback just triggers the event
            if 'save_checkpoint' in kwargs:
                kwargs['save_checkpoint'](checkpoint_path)


class WandBCallback(BaseCallback):
    """Callback for Weights & Biases logging."""

    def __init__(self, use_wandb: bool = True):
        self.use_wandb = use_wandb
        self.wandb = None

        if self.use_wandb:
            try:
                import wandb
                self.wandb = wandb
            except ImportError:
                print("⚠️  W&B not available")
                self.use_wandb = False

    def on_train_begin(self, **kwargs) -> None:
        if not self.use_wandb:
            return

        # W&B init is handled by trainer config
        print("📊 W&B logging enabled")

    def on_batch_end(self, state: TrainingState, **kwargs) -> None:
        if not self.use_wandb or not self.wandb:
            return

        # Use fractional_epoch if available, otherwise fall back to integer epoch
        epoch_value = state.fractional_epoch if state.fractional_epoch is not None else state.epoch

        log_dict = {
            'train/loss': state.loss,
            'train/learning_rate': state.learning_rate,
            'train/epoch': epoch_value,
        }

        if state.metrics:
            log_dict.update(state.metrics)

        try:
            self.wandb.log(log_dict, step=state.global_step)
        except Exception as e:
            print(f"⚠️  W&B logging failed: {e}")

    def on_validation_end(self, metrics: Dict[str, float], **kwargs) -> None:
        if not self.use_wandb or not self.wandb:
            return

        try:
            # Prefix validation metrics with 'val/'
            val_metrics = {f'val/{k}': v for k, v in metrics.items()}
            self.wandb.log(val_metrics)
        except Exception as e:
            print(f"⚠️  W&B validation logging failed: {e}")


class DiscordNotificationCallback(BaseCallback):
    """Callback for Discord notifications."""

    def __init__(self, alerter=None):
        self.alerter = alerter

    def on_train_begin(self, **kwargs) -> None:
        if self.alerter and self.alerter.enabled:
            model_info = kwargs.get('model_info', {})
            self.alerter.training_started(
                model_type=model_info.get('model_type', 'Unknown'),
                epochs=kwargs.get('num_epochs', 0),
                batch_size=kwargs.get('batch_size', 0),
                dataset_info=model_info.get('dataset_info', 'Unknown')
            )

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        if self.alerter and self.alerter.enabled:
            self.alerter.training_progress(
                epoch=epoch,
                total_epochs=kwargs.get('total_epochs', 0),
                loss=kwargs.get('loss', 0.0),
                lr=kwargs.get('learning_rate', 0.0),
                samples_per_sec=kwargs.get('samples_per_sec')
            )

    def on_train_end(self, **kwargs) -> None:
        if self.alerter and self.alerter.enabled:
            self.alerter.training_completed(
                duration_hours=kwargs.get('duration_hours', 0.0),
                final_metrics=kwargs.get('final_metrics', {})
            )


class CallbackManager:
    """Manages multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the manager."""
        self.callbacks.append(callback)

    def on_train_begin(self, **kwargs) -> None:
        """Trigger on_train_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_begin(**kwargs)

    def on_train_end(self, **kwargs) -> None:
        """Trigger on_train_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_train_end(**kwargs)

    def on_epoch_begin(self, epoch: int, **kwargs) -> None:
        """Trigger on_epoch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, **kwargs)

    def on_epoch_end(self, epoch: int, **kwargs) -> None:
        """Trigger on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, **kwargs)

    def on_batch_begin(self, batch_idx: int, **kwargs) -> None:
        """Trigger on_batch_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_begin(batch_idx, **kwargs)

    def on_batch_end(self, state: TrainingState, **kwargs) -> None:
        """Trigger on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(state, **kwargs)

    def on_validation_begin(self, **kwargs) -> None:
        """Trigger on_validation_begin for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_begin(**kwargs)

    def on_validation_end(self, metrics: Dict[str, float], **kwargs) -> None:
        """Trigger on_validation_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_validation_end(metrics, **kwargs)

    def should_stop_training(self) -> bool:
        """Check if any callback requests stopping training."""
        for callback in self.callbacks:
            if isinstance(callback, EarlyStoppingCallback) and callback.should_stop:
                return True
        return False


if __name__ == "__main__":
    print("🧪 Testing Callback System...")

    # Create callbacks
    progress_cb = ProgressCallback(log_every_n_steps=10)
    metrics_cb = MetricsCallback()
    early_stop_cb = EarlyStoppingCallback(patience=2, monitor='val_loss')

    # Create manager
    manager = CallbackManager([progress_cb, metrics_cb, early_stop_cb])

    # Simulate training
    manager.on_train_begin(num_epochs=3, batch_size=32)

    for epoch in range(3):
        manager.on_epoch_begin(epoch=epoch + 1)

        for step in range(20):
            state = TrainingState(
                epoch=epoch + 1,
                global_step=epoch * 20 + step + 1,
                batch_idx=step,
                total_batches=20,
                loss=1.0 - (epoch * 20 + step) * 0.01,
                learning_rate=5e-4
            )
            manager.on_batch_end(state)

        manager.on_epoch_end(epoch=epoch + 1, total_epochs=3, loss=0.5, learning_rate=5e-4)

        # Simulate validation
        manager.on_validation_begin()
        val_metrics = {'val_loss': 0.8 - epoch * 0.1}
        manager.on_validation_end(val_metrics)

        if manager.should_stop_training():
            print("🛑 Training stopped by callback")
            break

    manager.on_train_end(duration_hours=1.5, final_metrics={'train_loss': 0.3})

    print(f"\n📊 Average train loss: {metrics_cb.get_average_train_loss():.4f}")
    print(f"✅ Callback system test complete!")
