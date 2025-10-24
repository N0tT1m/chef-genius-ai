#!/usr/bin/env python3
"""
Modular Training System for Recipe Generation
Clean, extensible architecture for FLAN-T5 fine-tuning.
"""

from training.config import (
    CompleteTrainingConfig,
    TrainingConfig,
    DataConfig,
    OptimizationConfig,
    MonitoringConfig,
    EvaluationConfig,
    HardwareConfig,
    create_default_config,
)

from training.data_manager import DataManager, RecipeDataset
from training.metrics import MetricsCalculator, RecipeMetrics, RecipeQualityMetrics
from training.modular_trainer import ModularTrainer
from training.callbacks import (
    Callback,
    BaseCallback,
    ProgressCallback,
    MetricsCallback,
    EarlyStoppingCallback,
    WandBCallback,
    DiscordNotificationCallback,
    CallbackManager,
    TrainingState,
)

try:
    from training.lora_utils import LoRAManager, get_lora_target_modules_for_t5
    LORA_AVAILABLE = True
except ImportError:
    LORA_AVAILABLE = False

__version__ = "2.0.0"

__all__ = [
    # Config
    "CompleteTrainingConfig",
    "TrainingConfig",
    "DataConfig",
    "OptimizationConfig",
    "MonitoringConfig",
    "EvaluationConfig",
    "HardwareConfig",
    "create_default_config",
    # Data
    "DataManager",
    "RecipeDataset",
    # Metrics
    "MetricsCalculator",
    "RecipeMetrics",
    "RecipeQualityMetrics",
    # Trainer
    "ModularTrainer",
    # Callbacks
    "Callback",
    "BaseCallback",
    "ProgressCallback",
    "MetricsCallback",
    "EarlyStoppingCallback",
    "WandBCallback",
    "DiscordNotificationCallback",
    "CallbackManager",
    "TrainingState",
]

if LORA_AVAILABLE:
    __all__.extend(["LoRAManager", "get_lora_target_modules_for_t5"])
