#!/usr/bin/env python3
"""
LoRA (Low-Rank Adaptation) utilities for efficient fine-tuning.
Enables training 11B+ models with much less memory and faster iteration.
"""

from typing import Optional
import torch
from torch import nn

try:
    from peft import (
        LoraConfig,
        get_peft_model,
        TaskType,
        PeftModel,
        prepare_model_for_kbit_training
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️  PEFT library not installed. Install with: pip install peft")


class LoRAManager:
    """Manager for LoRA fine-tuning configuration and setup."""

    def __init__(
        self,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
        task_type: str = "SEQ_2_SEQ_LM"
    ):
        """
        Initialize LoRA manager.

        Args:
            lora_r: LoRA rank (typically 8, 16, or 32)
            lora_alpha: LoRA alpha parameter (typically 2x lora_r)
            lora_dropout: Dropout for LoRA layers
            target_modules: Which modules to apply LoRA to (default: q, v projection)
            task_type: Task type for PEFT (SEQ_2_SEQ_LM for T5)
        """
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library required for LoRA. Install with: pip install peft")

        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.target_modules = target_modules or ["q", "v"]
        self.task_type = task_type

        # Validate parameters
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        """Validate LoRA parameters."""
        if self.lora_r < 1:
            raise ValueError(f"lora_r must be >= 1, got {self.lora_r}")

        if self.lora_alpha < 1:
            raise ValueError(f"lora_alpha must be >= 1, got {self.lora_alpha}")

        if not 0 <= self.lora_dropout <= 1:
            raise ValueError(f"lora_dropout must be in [0, 1], got {self.lora_dropout}")

        if not self.target_modules:
            raise ValueError("target_modules cannot be empty")

        # Warn if alpha != 2*r (common convention)
        if self.lora_alpha != 2 * self.lora_r:
            print(f"⚠️  Note: lora_alpha={self.lora_alpha} != 2*lora_r={2*self.lora_r}")
            print(f"    Common practice is alpha=2*r, but this is fine if intentional.")

    def create_lora_config(self) -> LoraConfig:
        """Create LoRA configuration for PEFT."""
        task_type_map = {
            "SEQ_2_SEQ_LM": TaskType.SEQ_2_SEQ_LM,
            "CAUSAL_LM": TaskType.CAUSAL_LM,
        }

        return LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=self.target_modules,
            lora_dropout=self.lora_dropout,
            bias="none",
            task_type=task_type_map.get(self.task_type, TaskType.SEQ_2_SEQ_LM),
        )

    def apply_lora_to_model(self, model: nn.Module) -> PeftModel:
        """
        Apply LoRA to model and return PEFT model.

        Args:
            model: Base model to apply LoRA to

        Returns:
            PEFT model with LoRA adapters
        """
        lora_config = self.create_lora_config()

        print(f"\n🔧 Applying LoRA to model...")
        print(f"   Rank (r): {self.lora_r}")
        print(f"   Alpha: {self.lora_alpha}")
        print(f"   Dropout: {self.lora_dropout}")
        print(f"   Target modules: {', '.join(self.target_modules)}")

        # Get original parameter count
        original_params = sum(p.numel() for p in model.parameters())

        # Apply LoRA
        peft_model = get_peft_model(model, lora_config)

        # Get trainable parameter count
        trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in peft_model.parameters())

        print(f"\n📊 LoRA Parameter Statistics:")
        print(f"   Original parameters: {original_params:,}")
        print(f"   Trainable parameters: {trainable_params:,}")
        print(f"   All parameters: {all_params:,}")
        print(f"   Trainable ratio: {100 * trainable_params / all_params:.2f}%")
        print(f"   Parameter reduction: {100 * (1 - trainable_params / original_params):.1f}%")

        return peft_model

    @staticmethod
    def print_trainable_parameters(model: PeftModel) -> None:
        """Print trainable parameter statistics."""
        trainable_params = 0
        all_param = 0

        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()

        print(f"\n📊 Trainable Parameters:")
        print(f"   Trainable: {trainable_params:,} || Total: {all_param:,}")
        print(f"   Trainable %: {100 * trainable_params / all_param:.4f}%")

    @staticmethod
    def merge_and_unload(peft_model: PeftModel) -> nn.Module:
        """
        Merge LoRA weights into base model and return base model.
        Useful for inference or saving final model.

        Args:
            peft_model: PEFT model with LoRA adapters

        Returns:
            Base model with merged LoRA weights
        """
        print("\n🔀 Merging LoRA weights into base model...")
        merged_model = peft_model.merge_and_unload()
        print("✅ LoRA weights merged successfully")
        return merged_model

    @staticmethod
    def save_lora_adapters(peft_model: PeftModel, save_path: str) -> None:
        """
        Save only LoRA adapter weights (much smaller than full model).

        Args:
            peft_model: PEFT model with LoRA adapters
            save_path: Path to save adapters
        """
        print(f"\n💾 Saving LoRA adapters to: {save_path}")
        peft_model.save_pretrained(save_path)
        print("✅ LoRA adapters saved successfully")

    @staticmethod
    def load_lora_adapters(base_model: nn.Module, adapter_path: str) -> PeftModel:
        """
        Load LoRA adapters onto base model.

        Args:
            base_model: Base model
            adapter_path: Path to saved adapters

        Returns:
            PEFT model with loaded adapters
        """
        print(f"\n📂 Loading LoRA adapters from: {adapter_path}")
        peft_model = PeftModel.from_pretrained(base_model, adapter_path)
        print("✅ LoRA adapters loaded successfully")
        return peft_model


def get_lora_target_modules_for_t5() -> list[str]:
    """Get recommended target modules for T5 models."""
    # For T5, these are the attention projection layers
    return ["q", "v"]  # Query and Value projections


def get_lora_target_modules_for_llama() -> list[str]:
    """Get recommended target modules for LLaMA models."""
    return ["q_proj", "k_proj", "v_proj", "o_proj"]


def estimate_lora_memory_savings(
    model_params: int,
    lora_r: int,
    num_target_modules: int = 2
) -> dict[str, float]:
    """
    Estimate memory savings from using LoRA.

    Args:
        model_params: Number of parameters in base model
        lora_r: LoRA rank
        num_target_modules: Number of modules LoRA is applied to

    Returns:
        Dictionary with memory estimates
    """
    # Rough estimate: LoRA adds 2*d*r parameters per module
    # where d is hidden dimension (assume ~4096 for large models)
    d = 4096  # Rough estimate for large models

    lora_params = num_target_modules * 2 * d * lora_r
    reduction_ratio = lora_params / model_params

    return {
        'original_params': model_params,
        'lora_params': lora_params,
        'trainable_ratio': reduction_ratio,
        'parameter_reduction_percent': (1 - reduction_ratio) * 100,
        'estimated_memory_gb_full': model_params * 4 / 1e9,  # FP32
        'estimated_memory_gb_lora': lora_params * 4 / 1e9,
    }


if __name__ == "__main__":
    print("🧪 Testing LoRA utilities...")

    if not PEFT_AVAILABLE:
        print("❌ PEFT not available, skipping tests")
        exit(1)

    # Create LoRA manager
    lora_manager = LoRAManager(
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=get_lora_target_modules_for_t5()
    )

    print("✅ LoRA manager created successfully")

    # Estimate memory savings for FLAN-T5-XXL (11B parameters)
    estimates = estimate_lora_memory_savings(
        model_params=11_000_000_000,
        lora_r=16,
        num_target_modules=2
    )

    print(f"\n💾 Memory Estimates for FLAN-T5-XXL with LoRA:")
    print(f"   Original parameters: {estimates['original_params']:,}")
    print(f"   LoRA parameters: {estimates['lora_params']:,}")
    print(f"   Trainable ratio: {estimates['trainable_ratio']:.4%}")
    print(f"   Parameter reduction: {estimates['parameter_reduction_percent']:.1f}%")
    print(f"   Estimated full training memory: {estimates['estimated_memory_gb_full']:.1f} GB")
    print(f"   Estimated LoRA training memory: {estimates['estimated_memory_gb_lora']:.1f} GB")

    print("\n✅ LoRA utilities test complete!")
