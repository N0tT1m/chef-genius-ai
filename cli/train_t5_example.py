#!/usr/bin/env python3
"""
Example script for training T5/FLAN-T5 recipe generation model
Optimized for RTX 4090 with CUDA 12.6 and PyTorch 2.7.1
"""

import subprocess
import sys

def run_t5_training():
    """Run T5 training with optimal settings for RTX 4090."""
    
    # T5-Large training (770M parameters - good starting point)
    cmd_large = [
        sys.executable, "train_recipe_model.py",
        "--model-type", "t5",
        "--pretrained-model", "google/flan-t5-large",
        "--model-output", "./models/flan-t5-large-recipes",
        "--epochs", "10",
        "--batch-size", "8",  # Conservative for T5-Large
        "--learning-rate", "3e-4",  # Higher LR for fine-tuning
        "--max-length", "1024",  # Longer sequences for recipes
        "--gradient-accumulation-steps", "4",
        "--warmup-steps", "100",
        "--max-samples", "10000"  # For testing, remove for full training
    ]
    
    # T5-XL training (3B parameters - maximum for 4090)
    cmd_xl = [
        sys.executable, "train_recipe_model.py", 
        "--model-type", "t5",
        "--pretrained-model", "google/flan-t5-xl",
        "--model-output", "./models/flan-t5-xl-recipes",
        "--epochs", "5",  # Fewer epochs for larger model
        "--batch-size", "2",  # Smaller batch for XL model
        "--learning-rate", "1e-4",  # Lower LR for larger model
        "--max-length", "1024",
        "--gradient-accumulation-steps", "16",  # High accumulation to compensate
        "--warmup-steps", "50",
        "--max-samples", "5000"  # For testing
    ]
    
    print("Starting FLAN-T5-Large training (recommended first run)...")
    print("Command:", " ".join(cmd_large))
    
    try:
        subprocess.run(cmd_large, check=True)
        print("FLAN-T5-Large training completed successfully!")
        
        # Ask if user wants to train XL model
        response = input("\\nTrain FLAN-T5-XL (3B params) next? (y/n): ")
        if response.lower() == 'y':
            print("\\nStarting FLAN-T5-XL training...")
            print("Command:", " ".join(cmd_xl))
            subprocess.run(cmd_xl, check=True)
            print("FLAN-T5-XL training completed successfully!")
            
    except subprocess.CalledProcessError as e:
        print(f"Training failed with error: {e}")
        return False
    
    return True

def show_model_options():
    """Show available T5 model options for RTX 4090."""
    print("Available T5 models for RTX 4090 (24GB VRAM):")
    print()
    print("1. google/flan-t5-large (770M params)")
    print("   - Batch size: 8-16")
    print("   - Memory usage: ~8-12GB")
    print("   - Training time: Fast")
    print("   - Recommended for: First experiments")
    print()
    print("2. google/flan-t5-xl (3B params)")
    print("   - Batch size: 2-4") 
    print("   - Memory usage: ~18-22GB")
    print("   - Training time: Slower")
    print("   - Recommended for: Best quality")
    print()
    print("3. google/flan-t5-xxl (11B params)")
    print("   - Batch size: 1")
    print("   - Memory usage: ~20-24GB (tight fit)")
    print("   - Training time: Very slow")
    print("   - Recommended for: Inference only or LoRA fine-tuning")

if __name__ == "__main__":
    print("T5 Recipe Generation Training")
    print("=" * 40)
    
    show_model_options()
    print()
    
    response = input("Start training? (y/n): ")
    if response.lower() == 'y':
        run_t5_training()
    else:
        print("Training cancelled.")