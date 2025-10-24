#!/usr/bin/env python3
"""
Mistral Recipe Training Script
Optimized for Mistral-7B-Instruct-v0.1 with 4-bit quantization for RTX 4090
"""

import torch
import os
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from datasets import Dataset
import json
from pathlib import Path
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

class MistralRecipeDataset:
    """Recipe dataset class optimized for Mistral's chat template format."""
    
    def __init__(self, data_dir: str, tokenizer, max_length: int = 2048):
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
                        recipes.append(self.format_recipe_for_mistral(item))
                        
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
    
    def format_recipe_for_mistral(self, item):
        """Format recipe into Mistral's chat template format."""
        try:
            input_data = item['input_data']
            output_data = item['output_data']
            
            # Get ingredients
            ingredients = input_data.get('ingredients', [])
            if isinstance(ingredients, list):
                ingredients_text = ', '.join(str(ing).strip("'\"") for ing in ingredients[:10])
            else:
                ingredients_text = str(ingredients)
            
            # Get instructions and title
            instructions = output_data.get('instructions', [])
            title = output_data.get('title', 'Recipe')
            
            if instructions and isinstance(instructions, list):
                instructions_text = '\n'.join(f"{i+1}. {inst}" for i, inst in enumerate(instructions))
            else:
                instructions_text = "Follow standard cooking procedures"
            
            # Create chat format for Mistral
            user_message = f"Generate a complete recipe using these ingredients: {ingredients_text}"
            assistant_message = f"**{title}**\n\n**Ingredients:**\n{ingredients_text}\n\n**Instructions:**\n{instructions_text}"
            
            # Format as chat conversation
            messages = [
                {"role": "user", "content": user_message},
                {"role": "assistant", "content": assistant_message}
            ]
            
            # Apply chat template
            formatted_text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            return {'text': formatted_text}
            
        except Exception as e:
            # Fallback for malformed data
            fallback_messages = [
                {"role": "user", "content": "Generate a simple recipe"},
                {"role": "assistant", "content": "**Simple Recipe**\n\nIngredients: Basic ingredients\n\nInstructions:\n1. Prepare ingredients\n2. Cook according to standard methods"}
            ]
            
            return {'text': self.tokenizer.apply_chat_template(
                fallback_messages, 
                tokenize=False, 
                add_generation_prompt=False
            )}
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def create_training_dataset(data_dir: str, tokenizer):
    """Create training dataset from recipe data."""
    
    # Load recipe data
    recipe_dataset = MistralRecipeDataset(data_dir, tokenizer)
    
    # Convert to Hugging Face dataset
    data_dict = {
        'text': [item['text'] for item in recipe_dataset.data]
    }
    
    dataset = Dataset.from_dict(data_dict)
    
    def tokenize_function(examples):
        # Tokenize the entire conversation
        result = tokenizer(
            examples['text'],
            max_length=2048,
            padding=False,
            truncation=True,
            return_tensors=None
        )
        return result
    
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
    
    parser = argparse.ArgumentParser(description="Mistral Recipe Training")
    parser.add_argument("--output-dir", default="./models/mistral_recipe", help="Output directory")
    parser.add_argument("--data-dir", default="cli/data/datasets", help="Data directory")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--use-lora", action="store_true", help="Use LoRA fine-tuning")
    
    args = parser.parse_args()
    
    print("🍳 MISTRAL RECIPE TRAINING")
    print("="*40)
    print(f"📁 Output: {args.output_dir}")
    print(f"📂 Data: {args.data_dir}")
    print(f"📊 Epochs: {args.epochs}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"🔧 LoRA: {args.use_lora}")
    
    # Setup 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Setup model and tokenizer
    model_name = "mistralai/Mistral-7B-Instruct-v0.1"
    print(f"\n🔧 Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    # Set pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✅ Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Setup LoRA if requested
    if args.use_lora:
        print("\n🔧 Setting up LoRA...")
        model = prepare_model_for_kbit_training(model)
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # Create dataset
    print(f"\n📊 Creating dataset...")
    train_dataset = create_training_dataset(args.data_dir, tokenizer)
    print(f"✅ Dataset created: {len(train_dataset)} samples")
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        weight_decay=0.01,
        warmup_steps=100,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        dataloader_pin_memory=True,
        remove_unused_columns=False,
        report_to=[],
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        group_by_length=True,
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
    print(f"🔧 Optimizations:")
    print(f"  ✅ 4-bit quantization enabled")
    print(f"  ✅ Mistral chat template format")
    print(f"  ✅ Gradient checkpointing")
    print(f"  ✅ RTX 4090 optimized batch size")
    if args.use_lora:
        print(f"  ✅ LoRA fine-tuning enabled")
    
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