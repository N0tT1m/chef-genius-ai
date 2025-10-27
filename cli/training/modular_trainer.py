#!/usr/bin/env python3
"""
Modular Training System
Clean, extensible trainer that replaces the monolithic CompleteOptimizedTrainer.
"""

import os
import time
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, List
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup

from training.config import CompleteTrainingConfig
from training.data_manager import DataManager
from training.metrics import MetricsCalculator, RecipeMetrics
from training.callbacks import (
    CallbackManager,
    ProgressCallback,
    MetricsCallback,
    EarlyStoppingCallback,
    WandBCallback,
    DiscordNotificationCallback,
    TrainingState,
)
from training.lora_utils import LoRAManager
from training.checkpoint_utils import CheckpointManager


class ModularTrainer:
    """
    Clean, modular trainer for recipe generation models.
    Replaces the 1283-line monolithic trainer with composable components.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: CompleteTrainingConfig,
        discord_alerter=None,
    ):
        """
        Initialize the modular trainer.

        Args:
            model: Pre-trained model to fine-tune
            tokenizer: Model tokenizer
            config: Complete training configuration
            discord_alerter: Optional Discord alerter for notifications
        """
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Auto-adjust config based on model size
        num_params = sum(p.numel() for p in model.parameters())
        self.config.auto_adjust_for_model_size(num_params)

        # Apply LoRA if requested
        if self.config.training.use_lora:
            self._apply_lora()

        # Initialize components
        self.data_manager = DataManager(tokenizer, self.config.data)
        self.metrics_calculator = MetricsCalculator(
            compute_bleu=self.config.evaluation.compute_bleu,
            compute_rouge=self.config.evaluation.compute_rouge,
        )

        # Setup callbacks
        self.callback_manager = self._setup_callbacks(discord_alerter)

        # Training state
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None
        self.global_step = 0
        self.current_epoch = 0  # Track current epoch for proper resumption
        self.start_time: Optional[float] = None
        self.best_loss = float('inf')
        self.epoch_losses: List[float] = []

        # Setup hardware optimizations
        self._setup_hardware()

        # Move model to device
        self.model.to(self.device)

    def _apply_lora(self) -> None:
        """Apply LoRA to model if configured."""
        from training.lora_utils import LoRAManager, get_lora_target_modules_for_t5

        lora_manager = LoRAManager(
            lora_r=self.config.training.lora_r,
            lora_alpha=self.config.training.lora_alpha,
            lora_dropout=self.config.training.lora_dropout,
            target_modules=self.config.training.lora_target_modules or get_lora_target_modules_for_t5(),
        )

        self.model = lora_manager.apply_lora_to_model(self.model)
        print("✅ LoRA applied to model")

    def _setup_callbacks(self, discord_alerter) -> CallbackManager:
        """Setup training callbacks."""
        callbacks = []

        # Progress logging
        callbacks.append(ProgressCallback(log_every_n_steps=self.config.monitoring.log_every_n_steps))

        # Metrics tracking
        callbacks.append(MetricsCallback())

        # Early stopping
        if self.config.training.early_stopping_patience > 0:
            callbacks.append(EarlyStoppingCallback(
                patience=self.config.training.early_stopping_patience,
                min_delta=self.config.training.early_stopping_threshold,
            ))

        # W&B logging
        if self.config.monitoring.use_wandb:
            callbacks.append(WandBCallback(use_wandb=True))

        # Discord notifications
        if discord_alerter:
            callbacks.append(DiscordNotificationCallback(alerter=discord_alerter))

        return CallbackManager(callbacks)

    def _setup_hardware(self) -> None:
        """Setup hardware optimizations."""
        print("\n⚙️  Setting up hardware optimizations...")

        # CUDA optimizations
        if torch.cuda.is_available():
            # TF32 for tensor cores
            try:
                torch.backends.cudnn.conv.fp32_precision = 'tf32'
                torch.backends.cuda.matmul.fp32_precision = 'tf32'
                print("   ✅ TF32 enabled")
            except AttributeError:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                print("   ✅ TF32 enabled (legacy API)")

            torch.backends.cudnn.benchmark = True
            torch.cuda.empty_cache()

            # Memory optimization
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = (
                'expandable_segments:True,max_split_size_mb:128,'
                'roundup_power2_divisions:16,garbage_collection_threshold:0.7'
            )

            # Convert to BF16 if requested
            if self.config.training.use_bf16:
                if self.model.dtype != torch.bfloat16:
                    print("   Converting model to bfloat16...")
                    self.model = self.model.to(dtype=torch.bfloat16)
                    print("   ✅ Model converted to BF16")
                else:
                    print("   ✅ Model already in BF16")

        # CPU optimizations
        torch.set_num_threads(self.config.hardware.cpu_threads)
        os.environ['OMP_NUM_THREADS'] = str(self.config.hardware.cpu_threads)
        os.environ['MKL_NUM_THREADS'] = str(self.config.hardware.cpu_threads)

        # Gradient checkpointing
        if self.config.training.gradient_checkpointing:
            if hasattr(self.model, 'gradient_checkpointing_enable'):
                self.model.gradient_checkpointing_enable()
                if hasattr(self.model.config, 'use_cache'):
                    self.model.config.use_cache = False
                print("   ✅ Gradient checkpointing enabled")

        # Flash Attention
        if self.config.training.use_flash_attention:
            try:
                torch.backends.cuda.enable_flash_sdp(True)
                print("   ✅ Flash Attention enabled")
            except (AttributeError, Exception):
                print("   ⚠️  Flash Attention not available")

        # Torch compile
        if self.config.training.use_torch_compile and not self.config.training.use_lora:
            if self.config.training.disable_cudagraphs:
                os.environ['TORCH_COMPILE_DISABLE_CUDAGRAPHS'] = '1'

            try:
                import platform
                if platform.system() != 'Windows' and torch.cuda.is_available():
                    self.model = torch.compile(self.model, mode='reduce-overhead')
                    print("   ✅ Torch compile enabled")
            except Exception as e:
                print(f"   ⚠️  Torch compile failed: {e}")

        print("✅ Hardware optimizations complete\n")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer based on configuration."""
        if self.config.optimization.optimizer_type == "adamw_fused":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimization.learning_rate,
                weight_decay=self.config.optimization.weight_decay,
                betas=self.config.optimization.betas,
                eps=self.config.optimization.eps,
                fused=True,
            )
        else:
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.optimization.learning_rate,
                weight_decay=self.config.optimization.weight_decay,
                betas=self.config.optimization.betas,
                eps=self.config.optimization.eps,
            )

    def _create_scheduler(self, total_steps: int) -> torch.optim.lr_scheduler.LRScheduler:
        """Create learning rate scheduler."""
        if self.config.optimization.scheduler_type == "linear":
            return get_linear_schedule_with_warmup(
                self.optimizer,
                num_warmup_steps=self.config.optimization.warmup_steps,
                num_training_steps=total_steps,
            )
        elif self.config.optimization.scheduler_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
            )
        else:  # constant
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda step: 1.0,
            )

    def resume_from_checkpoint(self, checkpoint_path: str) -> None:
        """
        Resume training from a checkpoint.

        Args:
            checkpoint_path: Path to checkpoint directory
        """
        print(f"\n🔄 Resuming from checkpoint: {checkpoint_path}")

        # Load checkpoint state (model, optimizer, scheduler, RNG states)
        checkpoint_info = CheckpointManager.load_checkpoint(
            checkpoint_dir=checkpoint_path,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            device=str(self.device)
        )

        # Restore training state
        self.global_step = checkpoint_info.get('global_step', 0)
        self.current_epoch = checkpoint_info.get('epoch', 0)
        self.best_loss = checkpoint_info.get('best_loss', float('inf'))
        self.epoch_losses = checkpoint_info.get('epoch_losses', [])

        print(f"✅ Resumed from step {self.global_step}, epoch {self.current_epoch}")
        print(f"   Best loss: {self.best_loss:.4f}")
        print(f"   RNG states restored: {checkpoint_info.get('has_rng_states', False)}")

    def train(self, resume_checkpoint: Optional[str] = None) -> None:
        """
        Run training loop.

        Args:
            resume_checkpoint: Optional path to checkpoint to resume from
        """
        print("\n🚀 Starting Training")
        self.config.print_summary()

        # Load datasets
        self.data_manager.load_datasets()

        # Create dataloaders
        train_loader = self.data_manager.create_train_dataloader(
            batch_size=self.config.training.batch_size
        )
        val_loader = self.data_manager.create_val_dataloader(
            batch_size=self.config.training.batch_size
        )

        # Calculate total steps
        total_steps = (
            len(train_loader) * self.config.training.num_epochs
            // self.config.training.gradient_accumulation_steps
        )

        # Create optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler(total_steps)

        # Resume from checkpoint if provided
        if resume_checkpoint:
            self.resume_from_checkpoint(resume_checkpoint)

        print(f"\n📊 Training Plan:")
        print(f"   Total batches per epoch: {len(train_loader):,}")
        print(f"   Total training steps: {total_steps:,}")
        print(f"   Warmup steps: {self.config.optimization.warmup_steps:,}")
        print(f"   Validation batches: {len(val_loader):,}")
        if resume_checkpoint:
            print(f"   Resuming from step: {self.global_step:,}")

        # Training start callback
        self.callback_manager.on_train_begin(
            num_epochs=self.config.training.num_epochs,
            batch_size=self.config.training.batch_size,
            model_info={
                'model_type': type(self.model).__name__,
                'dataset_info': f"{self.data_manager.get_train_size():,} training samples"
            }
        )

        self.start_time = time.time()
        self.model.train()

        # Training loop
        try:
            # Start from current_epoch if resuming, otherwise from 0
            start_epoch = self.current_epoch
            for epoch in range(start_epoch, self.config.training.num_epochs):
                self.current_epoch = epoch
                self._train_epoch(epoch + 1, train_loader)

                # Validation
                if len(val_loader) > 0:
                    val_metrics = self._validate(val_loader)
                    self.callback_manager.on_validation_end(val_metrics)

                    # Check for early stopping
                    if self.callback_manager.should_stop_training():
                        print("🛑 Early stopping triggered")
                        break

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            raise
        finally:
            self._finalize_training()

    def _train_epoch(self, epoch: int, train_loader) -> None:
        """Train for one epoch."""
        self.callback_manager.on_epoch_begin(epoch=epoch)

        epoch_loss = 0.0
        batch_count = 0
        self.optimizer.zero_grad()

        # Calculate batches per epoch for proper resumption
        batches_per_epoch = len(train_loader)

        # When resuming, calculate which batch to start from based on global_step
        # epoch is 1-indexed in this function, current_epoch is 0-indexed
        steps_before_this_epoch = self.current_epoch * (batches_per_epoch // self.config.training.gradient_accumulation_steps)
        steps_in_this_epoch = max(0, self.global_step - steps_before_this_epoch)
        start_batch = steps_in_this_epoch * self.config.training.gradient_accumulation_steps

        if start_batch > 0:
            print(f"   📍 Resuming from batch {start_batch}/{batches_per_epoch} in epoch {epoch}")

        for batch_idx, batch in enumerate(train_loader):
            # Skip already-processed batches when resuming
            if batch_idx < start_batch:
                continue
            # Move batch to device
            batch = {k: v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v
                     for k, v in batch.items()}

            # Forward pass
            with torch.autocast(
                device_type='cuda',
                dtype=torch.bfloat16,
                enabled=self.config.training.use_bf16
            ):
                outputs = self.model(**{k: v for k, v in batch.items() if k != 'quality_scores'})
                loss = outputs.loss / self.config.training.gradient_accumulation_steps

            # Backward pass
            loss.backward()

            epoch_loss += loss.item()
            batch_count += 1

            # Optimizer step
            if batch_count % self.config.training.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.optimization.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                self.global_step += 1

                # Periodic cleanup
                if self.global_step % 5 == 0:
                    torch.cuda.empty_cache()

            # Batch end callback
            # Calculate fractional epoch based on global_step
            steps_per_epoch = batches_per_epoch / self.config.training.gradient_accumulation_steps
            fractional_epoch = self.global_step / steps_per_epoch if steps_per_epoch > 0 else 0.0

            state = TrainingState(
                epoch=epoch,
                global_step=self.global_step,
                batch_idx=batch_idx,
                total_batches=len(train_loader),
                loss=epoch_loss / batch_count,
                learning_rate=self.optimizer.param_groups[0]['lr'],
                fractional_epoch=fractional_epoch,
            )
            self.callback_manager.on_batch_end(state)

            # Checkpoint saving
            if (self.global_step % self.config.training.checkpoint_every_n_steps == 0
                and self.global_step > 0):
                self._save_checkpoint(self.global_step)

        # Epoch end
        avg_loss = epoch_loss / batch_count if batch_count > 0 else 0.0
        self.epoch_losses.append(avg_loss)

        # Update best loss
        if avg_loss < self.best_loss:
            self.best_loss = avg_loss
            print(f"   🎯 New best loss: {self.best_loss:.4f}")

        self.callback_manager.on_epoch_end(
            epoch=epoch,
            total_epochs=self.config.training.num_epochs,
            loss=avg_loss,
            learning_rate=self.optimizer.param_groups[0]['lr']
        )

    @torch.no_grad()
    def _validate(self, val_loader) -> Dict[str, float]:
        """Run validation loop."""
        self.callback_manager.on_validation_begin()

        self.model.eval()
        total_loss = 0.0
        batch_count = 0

        all_references = []
        all_hypotheses = []

        print("\n📊 Running validation...")

        for batch in val_loader:
            batch = {k: v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v
                     for k, v in batch.items()}

            # Calculate loss
            outputs = self.model(**{k: v for k, v in batch.items() if k != 'quality_scores'})
            total_loss += outputs.loss.item()
            batch_count += 1

            # Generate samples for metrics (first few batches only)
            if batch_count <= 3:
                generated = self.model.generate(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    max_new_tokens=128,
                    do_sample=False,
                    num_beams=1,
                )

                # Decode
                refs = self.tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
                hyps = self.tokenizer.batch_decode(generated, skip_special_tokens=True)

                all_references.extend(refs)
                all_hypotheses.extend(hyps)

        self.model.train()

        # Calculate metrics
        avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
        metrics_dict = {'val_loss': avg_loss}

        if all_references and all_hypotheses:
            recipe_metrics = self.metrics_calculator.calculate_all_metrics(
                references=all_references,
                hypotheses=all_hypotheses,
                loss=avg_loss
            )
            metrics_dict.update(recipe_metrics.to_dict())
            recipe_metrics.print_summary(prefix="   ")

        return metrics_dict

    def _save_checkpoint(self, step: int) -> None:
        """Save complete model checkpoint with full training state."""
        checkpoint_dir = Path(self.config.output_dir) / f"checkpoint-{step}"

        # Use CheckpointManager for complete state saving
        # Use self.current_epoch instead of calculating from steps
        CheckpointManager.save_checkpoint(
            checkpoint_dir=str(checkpoint_dir),
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            epoch=self.current_epoch,
            global_step=step,
            best_loss=self.best_loss,
            epoch_losses=self.epoch_losses,
            tokenizer=self.tokenizer
        )

        print(f"   💾 Complete checkpoint saved: {checkpoint_dir}")

    def _finalize_training(self) -> None:
        """Finalize training and save final model."""
        duration = time.time() - self.start_time if self.start_time else 0
        duration_hours = duration / 3600

        # Save final model
        print(f"\n💾 Saving final model to: {self.config.output_dir}")
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)

        if self.config.training.use_lora:
            self.model.save_pretrained(self.config.output_dir)
        else:
            self.model.save_pretrained(self.config.output_dir)

        self.tokenizer.save_pretrained(self.config.output_dir)

        # Training end callback
        self.callback_manager.on_train_end(
            duration_hours=duration_hours,
            final_metrics={'total_steps': self.global_step}
        )

        print(f"\n✅ Training complete! Duration: {duration_hours:.2f}h")
        print(f"   Final model saved to: {self.config.output_dir}")


if __name__ == "__main__":
    # Test the modular trainer
    print("🧪 Testing Modular Trainer...")

    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from training.config import create_default_config

    # Create config
    config = create_default_config()
    config.training.num_epochs = 1
    config.training.batch_size = 2
    config.experiment_name = "test_modular_trainer"

    # Load small model for testing
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small")

    # Create trainer
    trainer = ModularTrainer(model, tokenizer, config)

    print("\n✅ Modular trainer created successfully!")
    print("   Ready for training with train() method")
