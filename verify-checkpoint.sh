#!/bin/bash

# Chef Genius Checkpoint Verification Script

echo "🔍 Verifying checkpoint before Docker training..."

CHECKPOINT_DIR="./models/recipe_generation_flan-t5-large/checkpoint-1000"

if [ -d "$CHECKPOINT_DIR" ]; then
    echo "✅ Checkpoint directory found: $CHECKPOINT_DIR"
    
    # Check for essential files
    if [ -f "$CHECKPOINT_DIR/config.json" ]; then
        echo "✅ config.json found"
    else
        echo "❌ config.json missing"
        exit 1
    fi
    
    if [ -f "$CHECKPOINT_DIR/pytorch_model.bin" ] || [ -f "$CHECKPOINT_DIR/model.safetensors" ]; then
        echo "✅ Model weights found"
    else
        echo "❌ Model weights missing (pytorch_model.bin or model.safetensors)"
        exit 1
    fi
    
    if [ -f "$CHECKPOINT_DIR/tokenizer.json" ]; then
        echo "✅ tokenizer.json found"
    else
        echo "❌ tokenizer.json missing"
        exit 1
    fi
    
    echo ""
    echo "✅ Checkpoint verification passed!"
    echo "🚀 Ready to start Docker training from step 1,000"
    echo ""
    echo "Run: docker-train.bat (Windows) or ./train.sh (Linux/Mac)"
    
else
    echo "❌ Checkpoint directory not found: $CHECKPOINT_DIR"
    echo ""
    echo "Expected structure:"
    echo "$CHECKPOINT_DIR/"
    echo "├── config.json"
    echo "├── pytorch_model.bin (or model.safetensors)"
    echo "├── tokenizer.json"
    echo "├── tokenizer_config.json"
    echo "└── training_args.bin"
    echo ""
    echo "Please ensure your checkpoint exists before running Docker training."
    exit 1
fi