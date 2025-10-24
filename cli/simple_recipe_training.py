#!/usr/bin/env python3
"""
Simple Recipe Training Script - Fixed Version
Addresses all tokenization and configuration issues for FLAN-T5 recipe generation
"""

import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
import json
from pathlib import Path

class RecipeDataset:
    """Simple recipe dataset class that handles the data correctly."""
    
    def __init__(self, data_dir: str, tokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self.load_recipes(data_dir)
        
    def load_recipes(self, data_dir: str):
        """Load recipes from training.json files in all dataset directories."""
        recipes = []
        data_path = Path(data_dir)
        
        print(f"🔍 Loading recipes from {data_path}")
        
        # Find all training.json files
        training_files = list(data_path.glob("**/training.json"))
        print(f"📁 Found {len(training_files)} training files")
        
        for file_path in training_files[:5]:  # Limit to 5 datasets for testing
            print(f"📖 Loading {file_path.parent.name}")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                for item in data[:1000]:  # Limit samples per dataset
                    if self.is_valid_recipe(item):
                        recipes.append(self.format_recipe(item))
                        
            except Exception as e:
                print(f"⚠️  Error loading {file_path}: {e}")
                continue
                
        print(f"✅ Loaded {len(recipes)} valid recipes")
        return recipes
    
    def is_valid_recipe(self, item):
        """Check if recipe data is valid."""
        try:
            input_data = item.get('input_data', {})
            output_data = item.get('output_data', {})
            
            # Check for ingredients
            ingredients = input_data.get('ingredients', [])
            if not ingredients or len(ingredients) == 0:
                return False
                
            # Check for instructions
            instructions = output_data.get('instructions', [])
            title = output_data.get('title', '')
            
            if not instructions and not title:
                return False
                
            return True
            
        except Exception:
            return False
    
    def format_recipe(self, item):
        """Format recipe into input/target format for FLAN-T5."""
        try:
            input_data = item['input_data']
            output_data = item['output_data']
            
            # Get ingredients
            ingredients = input_data.get('ingredients', [])
            if isinstance(ingredients, list):
                ingredients_text = ', '.join(str(ing).strip("'\"") for ing in ingredients[:10])  # Limit ingredients
            else:
                ingredients_text = str(ingredients)
            
            # Get instructions
            instructions = output_data.get('instructions', [])
            title = output_data.get('title', '')
            
            if instructions and isinstance(instructions, list):
                instructions_text = ' '.join(str(inst) for inst in instructions)
            elif title:
                instructions_text = str(title)
            else:
                instructions_text = "Recipe instructions"
            
            # Create input prompt and target
            input_prompt = f"Generate a recipe using these ingredients: {ingredients_text}"
            target_recipe = f"Ingredients: {ingredients_text}\n\nInstructions: {instructions_text}"
            
            return {
                'input_text': input_prompt,
                'target_text': target_recipe
            }
            
        except Exception as e:
            # Fallback for malformed data
            return {
                'input_text': "Generate a simple recipe",
                'target_text': "Ingredients: Basic ingredients\n\nInstructions: Simple cooking steps"
            }
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_training_dataset(data_dir: str, tokenizer):
    """Create training dataset from recipe data."""
    
    # Load recipe data
    recipe_dataset = RecipeDataset(data_dir, tokenizer)
    
    # Convert to Hugging Face dataset
    data_dict = {
        'input_text': [item['input_text'] for item in recipe_dataset.data],
        'target_text': [item['target_text'] for item in recipe_dataset.data]
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    def tokenize_function(examples):
        # Tokenize inputs
        model_inputs = tokenizer(
            examples['input_text'],
            max_length=256,
            padding=False,
            truncation=True
        )
        
        # Tokenize targets
        labels = tokenizer(
            examples['target_text'],
            max_length=512,
            padding=False,
            truncation=True
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Recipe Training (Fixed)")
    parser.add_argument("--output-dir", default="./models/recipe_simple", help="Output directory")
    parser.add_argument("--data-dir", default="cli/data/datasets", help="Data directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    
    args = parser.parse_args()
    
    print("🍳 SIMPLE RECIPE TRAINING (FIXED)")
    print("="*40)
    print(f"📁 Output: {args.output_dir}")
    print(f"📂 Data: {args.data_dir}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    
    # Setup model and tokenizer
    model_name = "google/flan-t5-large"
    print(f"\n🔧 Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create dataset
    print(f"\n📊 Creating dataset...")
    train_dataset = create_training_dataset(args.data_dir, tokenizer)
    print(f"✅ Dataset created: {len(train_dataset)} samples")
    
    # Create data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        return_tensors="pt"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=5e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=50,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[],  # Disable wandb for now
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print(f"\n🚀 Starting training...")
    print(f"🔧 Fixed issues:")
    print(f"  ✅ Proper seq2seq tokenization")
    print(f"  ✅ Updated transformers API")
    print(f"  ✅ Valid training data format")
    print(f"  ✅ Correct input→target mapping")
    
    try:
        # Start training
        trainer.train()
        
        # Save final model
        trainer.save_model()
        tokenizer.save_pretrained(args.output_dir)
        
        print(f"\n✅ Training completed!")
        print(f"📁 Model saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        raise

if __name__ == "__main__":
    main()