#!/usr/bin/env python3
"""
Optimized Recipe Generation Training Script
Fixes all identified issues for high-quality recipe generation with FLAN-T5
"""

import os
import sys
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from complete_optimized_training import CompleteOptimizedTrainer

def setup_recipe_training():
    """Setup optimized recipe training configuration."""
    
    print("🍳 Setting up optimized recipe generation training...")
    print("="*60)
    
    # Load FLAN-T5 Large model optimized for recipe generation
    model_name = "google/flan-t5-large"
    print(f"📦 Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Set pad token if missing
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    return model, tokenizer

def create_optimized_training_args(output_dir: str):
    """Create optimized training arguments for recipe generation."""
    
    return TrainingArguments(
        output_dir=output_dir,
        
        # Training parameters optimized for recipe generation
        num_train_epochs=3,
        per_device_train_batch_size=4,  # Conservative for RTX 4090
        gradient_accumulation_steps=4,  # Effective batch size = 16
        
        # Learning rate optimized for FLAN-T5 fine-tuning
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=1000,
        
        # Memory optimizations
        fp16=False,  # Use bfloat16 instead
        bf16=True,   # Better for FLAN-T5
        dataloader_pin_memory=True,
        gradient_checkpointing=True,
        
        # Logging and saving
        logging_steps=50,
        save_steps=1000,
        save_total_limit=5,
        eval_strategy="steps",  # Updated parameter name
        eval_steps=1000,
        
        # Quality improvements
        label_smoothing_factor=0.1,  # Helps with overconfident predictions
        lr_scheduler_type="linear",
        
        # Performance
        dataloader_num_workers=8,
        remove_unused_columns=False,
        
        # Reproducibility
        seed=42,
        data_seed=42,
        
        # Prevent overfitting on recipe data
        max_grad_norm=1.0,
        
        # Report to W&B if available
        report_to=["wandb"] if "wandb" in sys.modules else [],
    )

def main():
    """Run optimized recipe training."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Recipe Generation Training")
    parser.add_argument("--output-dir", required=True, help="Output directory for trained model")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--discord-webhook", type=str, help="Discord webhook URL for notifications")
    parser.add_argument("--alert-phone", type=str, help="Phone number for SMS alerts")
    
    args = parser.parse_args()
    
    print("🍳 CHEF GENIUS OPTIMIZED RECIPE TRAINING")
    print("="*50)
    print(f"📁 Output: {args.output_dir}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔧 Model: FLAN-T5-Large with seq2seq optimization")
    print()
    
    # Setup model and tokenizer
    model, tokenizer = setup_recipe_training()
    
    # Create optimized trainer
    trainer = CompleteOptimizedTrainer(
        model=model,
        tokenizer=tokenizer,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        discord_webhook=args.discord_webhook,
        alert_phone=args.alert_phone,
        wandb_project="chef-genius-recipe-optimized",
        use_wandb=True,
        gradient_accumulation_steps=4,
        enable_mixed_precision=False,  # Use bfloat16 instead
        disable_compilation=False,     # Enable torch.compile for speed
        disable_cudagraphs=True,       # Safer for T5 models
        dataloader_num_workers=8
    )
    
    print("🚀 Starting optimized recipe training...")
    print("Key improvements:")
    print("  ✅ Proper seq2seq input→target formatting")
    print("  ✅ FLAN-T5 instruction-following prompts")
    print("  ✅ Optimized learning rate (5e-4)")
    print("  ✅ NaN loss detection and handling")
    print("  ✅ Memory-efficient bfloat16 training")
    print("  ✅ All datasets unified for maximum data")
    print()
    
    try:
        # Start training
        trainer.train_complete_optimized(epochs=args.epochs)
        print("✅ Training completed successfully!")
        
    except KeyboardInterrupt:
        print("\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()