#!/usr/bin/env python3
"""
Training Script V2 - Modular Training System
Clean entry point for recipe generation model training.

Usage:
    # Using YAML config (recommended)
    python train_v2.py --config configs/default_config.yaml

    # Override config values
    python train_v2.py --config configs/default_config.yaml --batch-size 48 --epochs 5

    # Using command-line args only
    python train_v2.py --model google/flan-t5-large --output ./models/my_model --epochs 3
"""

import os
import sys
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Add cli directory to path if needed
cli_dir = Path(__file__).parent
if str(cli_dir) not in sys.path:
    sys.path.insert(0, str(cli_dir))

from training import CompleteTrainingConfig, ModularTrainer, create_default_config


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Train recipe generation model with V2 modular system',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use default config
  python train_v2.py --config configs/default_config.yaml

  # Override specific parameters
  python train_v2.py --config configs/default_config.yaml --batch-size 48 --epochs 5

  # LoRA training for large model
  python train_v2.py --config configs/lora_config.yaml

  # Quick test
  python train_v2.py --config configs/fast_iteration_config.yaml
        """
    )

    # Config file
    parser.add_argument(
        '--config',
        type=str,
        help='Path to YAML configuration file'
    )

    # Model settings
    parser.add_argument(
        '--model',
        type=str,
        help='Model name or path (overrides config)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory (overrides config)'
    )
    parser.add_argument(
        '--resume',
        type=str,
        help='Resume from checkpoint path'
    )

    # Training parameters
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of epochs (overrides config)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size (overrides config)'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        help='Learning rate (overrides config)'
    )
    parser.add_argument(
        '--use-lora',
        action='store_true',
        help='Enable LoRA fine-tuning'
    )

    # Monitoring
    parser.add_argument(
        '--discord-webhook',
        type=str,
        help='Discord webhook URL for notifications'
    )
    parser.add_argument(
        '--alert-phone',
        type=str,
        help='Phone number for SMS alerts'
    )
    parser.add_argument(
        '--no-wandb',
        action='store_true',
        help='Disable W&B logging'
    )

    # Debugging
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode (fast iteration config)'
    )

    return parser.parse_args()


def load_config(args) -> CompleteTrainingConfig:
    """Load and configure training config from args."""

    # Load from YAML or create default
    if args.config:
        print(f"📄 Loading configuration from: {args.config}")
        config = CompleteTrainingConfig.from_yaml(args.config)
    elif args.debug:
        print("🐛 Debug mode: Using fast iteration config")
        config = CompleteTrainingConfig.from_yaml("configs/fast_iteration_config.yaml")
    else:
        print("📋 Using default configuration")
        config = create_default_config()

    # Apply command-line overrides
    if args.model:
        config.model_name = args.model
        print(f"   Overriding model: {args.model}")

    if args.output:
        config.output_dir = args.output
        print(f"   Overriding output dir: {args.output}")

    if args.resume:
        config.resume_from_checkpoint = args.resume
        print(f"   Resuming from: {args.resume}")

    if args.epochs:
        config.training.num_epochs = args.epochs
        print(f"   Overriding epochs: {args.epochs}")

    if args.batch_size:
        config.training.batch_size = args.batch_size
        print(f"   Overriding batch size: {args.batch_size}")

    if args.learning_rate:
        config.optimization.learning_rate = args.learning_rate
        print(f"   Overriding learning rate: {args.learning_rate}")

    if args.use_lora:
        config.training.use_lora = True
        print("   Enabling LoRA fine-tuning")

    if args.discord_webhook:
        config.monitoring.discord_webhook = args.discord_webhook
        print("   Discord notifications enabled")

    if args.alert_phone:
        config.monitoring.alert_phone = args.alert_phone
        print("   SMS notifications enabled")

    if args.no_wandb:
        config.monitoring.use_wandb = False
        print("   W&B logging disabled")

    return config


def setup_discord_alerter(config: CompleteTrainingConfig):
    """Setup Discord alerter if configured."""
    if config.monitoring.discord_webhook or config.monitoring.alert_phone:
        # Import from old system (reuse existing alerter)
        try:
            from complete_optimized_training import DiscordAlerter
            return DiscordAlerter(
                webhook_url=config.monitoring.discord_webhook,
                phone_number=config.monitoring.alert_phone
            )
        except ImportError:
            print("⚠️  Discord alerter not available")
            return None
    return None


def main():
    """Main training entry point."""
    print("=" * 60)
    print("🍳 Chef Genius - Recipe Generation Training V2")
    print("=" * 60)

    # Parse arguments
    args = parse_args()

    # Load configuration
    config = load_config(args)

    # Setup Discord/SMS notifications
    alerter = setup_discord_alerter(config)

    # Print configuration summary
    config.print_summary()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("\n⚠️  WARNING: CUDA not available, training will be very slow!")
        print("   Consider using a GPU for reasonable training times.")
        response = input("Continue anyway? [y/N]: ")
        if response.lower() != 'y':
            print("Exiting...")
            return

    # Load model and tokenizer
    print(f"\n🔄 Loading model and tokenizer...")
    print(f"   Model: {config.model_name}")

    try:
        if config.resume_from_checkpoint:
            print(f"   Resuming from: {config.resume_from_checkpoint}")
            tokenizer = AutoTokenizer.from_pretrained(config.resume_from_checkpoint)
            model = AutoModelForSeq2SeqLM.from_pretrained(config.resume_from_checkpoint)
        else:
            tokenizer = AutoTokenizer.from_pretrained(config.model_name)

            # Load with optimal settings
            load_kwargs = {
                'torch_dtype': torch.bfloat16 if config.training.use_bf16 else torch.float32,
            }

            # Try Flash Attention 2
            if config.training.use_flash_attention:
                try:
                    load_kwargs['attn_implementation'] = 'flash_attention_2'
                    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, **load_kwargs)
                    print("   ✅ Flash Attention 2 enabled")
                except Exception as e:
                    print(f"   ⚠️  Flash Attention 2 not available: {e}")
                    load_kwargs.pop('attn_implementation')
                    model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, **load_kwargs)
            else:
                model = AutoModelForSeq2SeqLM.from_pretrained(config.model_name, **load_kwargs)

        # Setup tokenizer
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            print("   Set pad_token = eos_token")

        print(f"✅ Model loaded successfully")
        print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")

    except Exception as e:
        print(f"\n❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Create trainer
    print(f"\n🎯 Creating modular trainer...")
    try:
        trainer = ModularTrainer(
            model=model,
            tokenizer=tokenizer,
            config=config,
            discord_alerter=alerter
        )
        print("✅ Trainer created successfully")
    except Exception as e:
        print(f"\n❌ Failed to create trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Start training
    print(f"\n🚀 Starting training...")
    try:
        trainer.train()
        print(f"\n✅ Training completed successfully!")
        print(f"   Model saved to: {config.output_dir}")
        return 0

    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
        if alerter:
            alerter.training_crashed("Training interrupted by user (Ctrl+C)")
        return 130

    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        if alerter:
            alerter.training_crashed(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
