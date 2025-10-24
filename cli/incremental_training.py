#!/usr/bin/env python3
"""
Incremental Training Pipeline for Recipe Model
Enables continued training on new recipes while preventing catastrophic forgetting
"""

import os
import json
import shutil
from pathlib import Path
from datetime import datetime
import logging
from typing import Dict, List, Optional, Union
import torch
from transformers import (
    GPT2LMHeadModel, GPT2Tokenizer, TrainingArguments, 
    Trainer, DataCollatorForLanguageModeling
)
from datasets import Dataset, concatenate_datasets
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IncrementalTrainer:
    """Handles incremental training with checkpoint versioning and forgetting prevention."""
    
    def __init__(self, base_model_path: str, training_config: Dict):
        self.base_model_path = Path(base_model_path)
        self.config = training_config
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def load_existing_model(self):
        """Load the existing trained model."""
        logger.info(f"Loading model from {self.base_model_path}")
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.base_model_path)
        self.model = GPT2LMHeadModel.from_pretrained(self.base_model_path)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def create_checkpoint_backup(self, version: str) -> str:
        """Create versioned backup of current model."""
        backup_dir = self.base_model_path.parent / f"backups/v{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model files
        for file_path in self.base_model_path.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, backup_dir / file_path.name)
                
        logger.info(f"Created backup at {backup_dir}")
        return str(backup_dir)
        
    def validate_new_recipes(self, new_data: List[Dict]) -> List[Dict]:
        """Validate and clean new recipe data."""
        validated = []
        
        for recipe in new_data:
            # Check required fields
            if not all(key in recipe for key in ['title', 'ingredients', 'instructions']):
                logger.warning(f"Skipping recipe missing required fields: {recipe.get('title', 'Unknown')}")
                continue
                
            # Basic content validation
            if len(recipe['ingredients']) < 2:
                logger.warning(f"Skipping recipe with too few ingredients: {recipe['title']}")
                continue
                
            if len(recipe['instructions']) < 20:
                logger.warning(f"Skipping recipe with insufficient instructions: {recipe['title']}")
                continue
                
            validated.append(recipe)
            
        logger.info(f"Validated {len(validated)} out of {len(new_data)} new recipes")
        return validated
        
    def format_recipe_for_training(self, recipe: Dict) -> str:
        """Format recipe data into training text."""
        ingredients_text = " | ".join(recipe['ingredients']) if isinstance(recipe['ingredients'], list) else recipe['ingredients']
        
        formatted = f"<TITLE>{recipe['title']}</TITLE>"
        
        if 'cuisine' in recipe:
            formatted += f"<CUISINE>{recipe['cuisine']}</CUISINE>"
            
        formatted += f"<INGREDIENTS>{ingredients_text}</INGREDIENTS>"
        formatted += f"<INSTRUCTIONS>{recipe['instructions']}</INSTRUCTIONS>"
        
        return formatted
        
    def prepare_incremental_dataset(self, new_recipes: List[Dict], existing_sample_ratio: float = 0.3) -> Dataset:
        """Prepare dataset mixing new recipes with existing data sample."""
        # Format new recipes
        new_texts = [self.format_recipe_for_training(recipe) for recipe in new_recipes]
        
        # Load sample of existing training data to prevent forgetting
        existing_data = []
        if existing_sample_ratio > 0:
            try:
                # Try to load from cached training data
                training_data_path = self.base_model_path.parent / "training_data_cache.json"
                if training_data_path.exists():
                    with open(training_data_path, 'r') as f:
                        cached_data = json.load(f)
                    
                    # Sample existing data
                    import random
                    sample_size = min(int(len(new_texts) / existing_sample_ratio * (1 - existing_sample_ratio)), len(cached_data))
                    existing_data = random.sample(cached_data, sample_size)
                    logger.info(f"Loaded {len(existing_data)} existing recipes to prevent forgetting")
                    
            except Exception as e:
                logger.warning(f"Could not load existing training data: {e}")
        
        # Combine datasets
        all_texts = new_texts + existing_data
        
        # Tokenize
        tokenized = self.tokenizer(
            all_texts,
            truncation=True,
            padding=True,
            max_length=self.config.get('max_length', 512),
            return_tensors='pt'
        )
        
        return Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()
        })
        
    def train_incremental(self, new_recipes: List[Dict], output_dir: str, resume_from_checkpoint: bool = True):
        """Perform incremental training on new recipes."""
        
        # Validate input
        validated_recipes = self.validate_new_recipes(new_recipes)
        if not validated_recipes:
            logger.error("No valid recipes to train on")
            return
            
        # Create backup
        backup_path = self.create_checkpoint_backup("pre_incremental")
        
        # Prepare dataset
        train_dataset = self.prepare_incremental_dataset(validated_recipes)
        
        # Setup training arguments for incremental learning
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=self.config.get('incremental_epochs', 2),  # Fewer epochs for incremental
            per_device_train_batch_size=self.config.get('batch_size', 4),
            learning_rate=self.config.get('incremental_lr', 1e-5),  # Lower LR for incremental
            warmup_steps=self.config.get('warmup_steps', 100),
            save_steps=self.config.get('save_steps', 500),
            save_total_limit=5,
            logging_steps=50,
            evaluation_strategy="no",
            load_best_model_at_end=False,
            gradient_accumulation_steps=self.config.get('gradient_accumulation_steps', 2),
            dataloader_num_workers=0,
            report_to=["mlflow"] if self.config.get('use_mlflow', False) else [],
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Train
        logger.info(f"Starting incremental training on {len(validated_recipes)} new recipes")
        try:
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            
            # Save the updated model
            self.trainer.save_model()
            self.tokenizer.save_pretrained(output_dir)
            
            # Update training cache with new recipes
            self._update_training_cache(validated_recipes)
            
            logger.info(f"Incremental training completed. Model saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Restore from backup
            logger.info(f"Restoring model from backup: {backup_path}")
            self._restore_from_backup(backup_path)
            raise
            
    def _update_training_cache(self, new_recipes: List[Dict]):
        """Update the training data cache with new recipes."""
        cache_path = self.base_model_path.parent / "training_data_cache.json"
        
        # Load existing cache
        existing_cache = []
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    existing_cache = json.load(f)
            except Exception as e:
                logger.warning(f"Could not load existing cache: {e}")
        
        # Add new formatted recipes
        new_formatted = [self.format_recipe_for_training(recipe) for recipe in new_recipes]
        updated_cache = existing_cache + new_formatted
        
        # Limit cache size to prevent it from growing too large
        max_cache_size = self.config.get('max_cache_size', 10000)
        if len(updated_cache) > max_cache_size:
            updated_cache = updated_cache[-max_cache_size:]
            
        # Save updated cache
        with open(cache_path, 'w') as f:
            json.dump(updated_cache, f)
            
        logger.info(f"Updated training cache with {len(new_formatted)} new recipes")
        
    def _restore_from_backup(self, backup_path: str):
        """Restore model from backup."""
        backup_dir = Path(backup_path)
        
        for file_path in backup_dir.glob("*"):
            if file_path.is_file():
                shutil.copy2(file_path, self.base_model_path / file_path.name)
                
        logger.info(f"Model restored from backup: {backup_path}")


def load_new_recipes_from_csv(csv_path: str) -> List[Dict]:
    """Load new recipes from CSV file."""
    df = pd.read_csv(csv_path)
    
    recipes = []
    for _, row in df.iterrows():
        recipe = {
            'title': row.get('title', row.get('name', 'Unknown Recipe')),
            'ingredients': row.get('ingredients', '').split('|') if '|' in str(row.get('ingredients', '')) else [row.get('ingredients', '')],
            'instructions': row.get('instructions', row.get('directions', '')),
        }
        
        # Add optional fields
        if 'cuisine' in row and pd.notna(row['cuisine']):
            recipe['cuisine'] = row['cuisine']
            
        recipes.append(recipe)
        
    return recipes


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Incremental training for recipe model")
    parser.add_argument("--base-model", required=True, help="Path to existing trained model")
    parser.add_argument("--new-data", required=True, help="Path to CSV file with new recipes")
    parser.add_argument("--output-dir", required=True, help="Output directory for updated model")
    parser.add_argument("--epochs", type=int, default=2, help="Number of incremental training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate for incremental training")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--existing-ratio", type=float, default=0.3, help="Ratio of existing data to include")
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'incremental_epochs': args.epochs,
        'incremental_lr': args.learning_rate,
        'batch_size': args.batch_size,
        'max_length': 512,
        'warmup_steps': 100,
        'save_steps': 500,
        'gradient_accumulation_steps': 2,
    }
    
    # Initialize trainer
    trainer = IncrementalTrainer(args.base_model, config)
    trainer.load_existing_model()
    
    # Load new recipes
    new_recipes = load_new_recipes_from_csv(args.new_data)
    logger.info(f"Loaded {len(new_recipes)} new recipes from {args.new_data}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Perform incremental training
    trainer.train_incremental(new_recipes, args.output_dir)


if __name__ == "__main__":
    main()