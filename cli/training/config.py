#!/usr/bin/env python3
"""
Training Configuration with Pydantic validation and YAML support.
Replaces hardcoded parameters with validated, version-controlled configs.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Literal
import yaml
import torch
from pydantic import BaseModel, Field, field_validator, model_validator


class HardwareConfig(BaseModel):
    """Hardware-specific configuration with auto-detection."""

    cpu_cores: int = Field(default=12, description="Number of CPU cores")
    cpu_threads: int = Field(default=24, description="Number of CPU threads")
    gpu_vram_gb: float = Field(default=24.0, description="GPU VRAM in GB")
    system_ram_gb: int = Field(default=48, description="System RAM in GB")

    @classmethod
    def auto_detect(cls) -> "HardwareConfig":
        """Auto-detect hardware capabilities."""
        import os

        cpu_count = os.cpu_count() or 12

        if torch.cuda.is_available():
            gpu_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            gpu_vram = 0.0

        return cls(
            cpu_cores=cpu_count,
            cpu_threads=cpu_count,
            gpu_vram_gb=gpu_vram,
            system_ram_gb=48  # Default, hard to detect
        )


class DataConfig(BaseModel):
    """Data loading and preprocessing configuration."""

    validated_data_dir: str = Field(default="./cli/validated_datasets")
    min_quality_score: float = Field(default=0.6, ge=0.0, le=1.0)
    max_input_length: int = Field(default=256, gt=0)
    max_target_length: int = Field(default=512, gt=0)
    train_split: float = Field(default=0.9, gt=0.0, lt=1.0)
    shuffle: bool = Field(default=True)
    num_workers: int = Field(default=4, ge=0)
    use_rust_loader: bool = Field(default=True)

    @field_validator('train_split')
    @classmethod
    def validate_train_split(cls, v: float) -> float:
        """Ensure train split is reasonable."""
        if v < 0.5 or v > 0.99:
            raise ValueError("train_split must be between 0.5 and 0.99")
        return v


class OptimizationConfig(BaseModel):
    """Optimization settings (optimizer, scheduler, etc)."""

    optimizer_type: Literal["adamw", "adamw_fused", "adafactor"] = Field(default="adamw")
    learning_rate: float = Field(default=5e-4, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, gt=0.0)
    warmup_steps: int = Field(default=1000, ge=0)
    scheduler_type: Literal["linear", "cosine", "constant"] = Field(default="linear")
    betas: tuple[float, float] = Field(default=(0.9, 0.999))
    eps: float = Field(default=1e-8, gt=0.0)

    @field_validator('learning_rate')
    @classmethod
    def validate_lr(cls, v: float) -> float:
        """Warn if learning rate is unusual."""
        if v > 1e-3:
            print(f"⚠️  Warning: Learning rate {v} is quite high. Typical range: 1e-5 to 5e-4")
        return v


class TrainingConfig(BaseModel):
    """Core training configuration."""

    # Basic training params
    num_epochs: int = Field(default=3, ge=1)
    batch_size: int = Field(default=32, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)

    # Mixed precision
    use_bf16: bool = Field(default=True)
    use_fp16: bool = Field(default=False)

    # Memory optimizations
    gradient_checkpointing: bool = Field(default=True)
    use_flash_attention: bool = Field(default=True)

    # Compilation
    use_torch_compile: bool = Field(default=True)
    disable_cudagraphs: bool = Field(default=True)

    # Checkpointing
    checkpoint_every_n_steps: int = Field(default=1000, ge=1)
    keep_best_n_checkpoints: int = Field(default=3, ge=1)

    # Early stopping
    early_stopping_patience: int = Field(default=3, ge=1)
    early_stopping_threshold: float = Field(default=0.0001, ge=0.0)

    # LoRA settings (optional)
    use_lora: bool = Field(default=False)
    lora_r: int = Field(default=16, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    lora_target_modules: list[str] = Field(default_factory=lambda: ["q", "v"])

    @model_validator(mode='after')
    def validate_precision(self) -> 'TrainingConfig':
        """Ensure only one precision mode is enabled."""
        if self.use_bf16 and self.use_fp16:
            raise ValueError("Cannot use both BF16 and FP16. Choose one.")
        return self

    @property
    def effective_batch_size(self) -> int:
        """Calculate effective batch size with gradient accumulation."""
        return self.batch_size * self.gradient_accumulation_steps


class MonitoringConfig(BaseModel):
    """Monitoring and logging configuration."""

    # W&B
    use_wandb: bool = Field(default=True)
    wandb_project: str = Field(default="chef-genius-optimized")
    wandb_entity: Optional[str] = Field(default=None)

    # Discord notifications
    discord_webhook: Optional[str] = Field(default=None)
    alert_phone: Optional[str] = Field(default=None)

    # Logging
    log_every_n_steps: int = Field(default=50, ge=1)
    eval_every_n_steps: int = Field(default=1000, ge=1)
    generate_samples_every_n_steps: int = Field(default=1000, ge=1)

    # System monitoring
    monitor_system_metrics: bool = Field(default=True)
    monitor_data_pipeline: bool = Field(default=True)


class EvaluationConfig(BaseModel):
    """Evaluation metrics configuration."""

    # Metrics to compute
    compute_bleu: bool = Field(default=True)
    compute_rouge: bool = Field(default=True)
    compute_perplexity: bool = Field(default=True)

    # Recipe-specific metrics
    compute_ingredient_coherence: bool = Field(default=True)
    compute_instruction_quality: bool = Field(default=True)

    # Validation prompts
    validation_prompts: list[str] = Field(default_factory=lambda: [
        "Create a simple pasta dish with tomatoes and basil.",
        "Make a healthy breakfast with eggs and vegetables.",
        "Design a quick 15-minute dinner recipe.",
        "Bake chocolate chip cookies that are crispy outside and chewy inside.",
        "Prepare a vegetarian curry with chickpeas and spinach."
    ])


class CompleteTrainingConfig(BaseModel):
    """Complete training configuration combining all sub-configs."""

    # Model settings
    model_name: str = Field(default="google/flan-t5-large")
    output_dir: str = Field(default="./optimized_model")
    resume_from_checkpoint: Optional[str] = Field(default=None)

    # Sub-configurations
    hardware: HardwareConfig = Field(default_factory=HardwareConfig.auto_detect)
    data: DataConfig = Field(default_factory=DataConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)

    # Metadata
    experiment_name: str = Field(default="default_experiment")
    description: str = Field(default="")
    tags: list[str] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CompleteTrainingConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = self.model_dump()
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    @classmethod
    def from_args(cls, args) -> "CompleteTrainingConfig":
        """Create config from argparse arguments."""
        config = cls()

        # Override with command line args
        if hasattr(args, 'batch_size') and args.batch_size:
            config.training.batch_size = args.batch_size
        if hasattr(args, 'epochs') and args.epochs:
            config.training.num_epochs = args.epochs
        if hasattr(args, 'learning_rate') and args.learning_rate:
            config.optimization.learning_rate = args.learning_rate
        if hasattr(args, 'model_output') and args.model_output:
            config.output_dir = args.model_output
        if hasattr(args, 'pretrained_model') and args.pretrained_model:
            config.model_name = args.pretrained_model
        if hasattr(args, 'discord_webhook') and args.discord_webhook:
            config.monitoring.discord_webhook = args.discord_webhook
        if hasattr(args, 'alert_phone') and args.alert_phone:
            config.monitoring.alert_phone = args.alert_phone
        if hasattr(args, 'use_lora') and args.use_lora:
            config.training.use_lora = args.use_lora

        return config

    def auto_adjust_for_model_size(self, num_parameters: int) -> None:
        """Auto-adjust hyperparameters based on model size."""
        # Adjust learning rate for larger models
        if num_parameters > 10e9:  # > 10B parameters (XXL)
            self.optimization.learning_rate = 1e-4
            self.training.gradient_accumulation_steps = max(16, self.training.gradient_accumulation_steps)
            print(f"🎯 Detected XXL model ({num_parameters/1e9:.1f}B params) - adjusted LR to 1e-4")
        elif num_parameters > 3e9:  # > 3B parameters (XL)
            self.optimization.learning_rate = 3e-4
            self.training.gradient_accumulation_steps = max(8, self.training.gradient_accumulation_steps)
            print(f"🎯 Detected XL model ({num_parameters/1e9:.1f}B params) - adjusted LR to 3e-4")

    def auto_adjust_for_gpu(self) -> None:
        """Auto-adjust batch size and settings based on GPU VRAM."""
        vram = self.hardware.gpu_vram_gb

        if vram >= 30:  # RTX 5090 or better
            if self.training.batch_size < 48:
                self.training.batch_size = 48
                print(f"🎯 Detected {vram:.1f}GB VRAM - adjusted batch size to 48")
        elif vram >= 22:  # RTX 4090
            if self.training.batch_size < 32:
                self.training.batch_size = 32
                print(f"🎯 Detected {vram:.1f}GB VRAM - adjusted batch size to 32")
        elif vram >= 10:  # RTX 3080/3090
            if self.training.batch_size < 16:
                self.training.batch_size = 16
                self.training.gradient_checkpointing = True
                print(f"🎯 Detected {vram:.1f}GB VRAM - adjusted batch size to 16 with gradient checkpointing")

    def print_summary(self) -> None:
        """Print a human-readable summary of the configuration."""
        print("\n" + "="*60)
        print("📋 TRAINING CONFIGURATION SUMMARY")
        print("="*60)

        print(f"\n🔬 Experiment: {self.experiment_name}")
        if self.description:
            print(f"📝 Description: {self.description}")
        if self.tags:
            print(f"🏷️  Tags: {', '.join(self.tags)}")

        print(f"\n🤖 Model:")
        print(f"   Name: {self.model_name}")
        print(f"   Output: {self.output_dir}")
        if self.resume_from_checkpoint:
            print(f"   Resume from: {self.resume_from_checkpoint}")

        print(f"\n💾 Hardware:")
        print(f"   CPU: {self.hardware.cpu_cores} cores / {self.hardware.cpu_threads} threads")
        print(f"   GPU VRAM: {self.hardware.gpu_vram_gb:.1f} GB")
        print(f"   System RAM: {self.hardware.system_ram_gb} GB")

        print(f"\n📊 Training:")
        print(f"   Epochs: {self.training.num_epochs}")
        print(f"   Batch size: {self.training.batch_size}")
        print(f"   Gradient accumulation: {self.training.gradient_accumulation_steps}")
        print(f"   Effective batch size: {self.training.effective_batch_size}")
        print(f"   Precision: {'BF16' if self.training.use_bf16 else ('FP16' if self.training.use_fp16 else 'FP32')}")
        print(f"   Gradient checkpointing: {self.training.gradient_checkpointing}")
        print(f"   Flash Attention: {self.training.use_flash_attention}")
        print(f"   Torch compile: {self.training.use_torch_compile}")

        if self.training.use_lora:
            print(f"\n🔧 LoRA:")
            print(f"   Rank (r): {self.training.lora_r}")
            print(f"   Alpha: {self.training.lora_alpha}")
            print(f"   Dropout: {self.training.lora_dropout}")
            print(f"   Target modules: {', '.join(self.training.lora_target_modules)}")

        print(f"\n⚡ Optimization:")
        print(f"   Optimizer: {self.optimization.optimizer_type}")
        print(f"   Learning rate: {self.optimization.learning_rate}")
        print(f"   Weight decay: {self.optimization.weight_decay}")
        print(f"   Warmup steps: {self.optimization.warmup_steps}")
        print(f"   Scheduler: {self.optimization.scheduler_type}")
        print(f"   Max grad norm: {self.optimization.max_grad_norm}")

        print(f"\n📂 Data:")
        print(f"   Data dir: {self.data.validated_data_dir}")
        print(f"   Min quality score: {self.data.min_quality_score}")
        print(f"   Train/val split: {self.data.train_split:.0%} / {(1-self.data.train_split):.0%}")
        print(f"   Max input length: {self.data.max_input_length}")
        print(f"   Max target length: {self.data.max_target_length}")
        print(f"   Use Rust loader: {self.data.use_rust_loader}")

        print(f"\n📈 Monitoring:")
        print(f"   W&B: {self.monitoring.use_wandb}")
        if self.monitoring.use_wandb:
            print(f"   W&B project: {self.monitoring.wandb_project}")
        print(f"   Discord alerts: {'✅' if self.monitoring.discord_webhook else '❌'}")
        print(f"   SMS alerts: {'✅' if self.monitoring.alert_phone else '❌'}")

        print(f"\n📊 Evaluation:")
        metrics = []
        if self.evaluation.compute_bleu:
            metrics.append("BLEU")
        if self.evaluation.compute_rouge:
            metrics.append("ROUGE")
        if self.evaluation.compute_perplexity:
            metrics.append("Perplexity")
        if self.evaluation.compute_ingredient_coherence:
            metrics.append("Ingredient Coherence")
        if self.evaluation.compute_instruction_quality:
            metrics.append("Instruction Quality")
        print(f"   Metrics: {', '.join(metrics)}")
        print(f"   Validation prompts: {len(self.evaluation.validation_prompts)}")

        print("\n" + "="*60 + "\n")


def create_default_config() -> CompleteTrainingConfig:
    """Create a default configuration with auto-detected hardware."""
    config = CompleteTrainingConfig()
    config.auto_adjust_for_gpu()
    return config


if __name__ == "__main__":
    # Test configuration system
    config = create_default_config()
    config.experiment_name = "test_experiment"
    config.description = "Testing the new configuration system"
    config.tags = ["test", "modular", "refactor"]

    config.print_summary()

    # Save to YAML
    config.to_yaml("/tmp/test_config.yaml")
    print("✅ Saved configuration to /tmp/test_config.yaml")

    # Load from YAML
    loaded_config = CompleteTrainingConfig.from_yaml("/tmp/test_config.yaml")
    print("✅ Loaded configuration from YAML")

    print("\n✅ Configuration system test complete!")
