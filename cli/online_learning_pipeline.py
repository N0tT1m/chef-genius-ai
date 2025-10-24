#!/usr/bin/env python3
"""
Online Learning Pipeline for Recipe Model
Enables continuous learning from new recipes with validation and monitoring
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
import pandas as pd
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from incremental_training import IncrementalTrainer, load_new_recipes_from_csv
from model_versioning import ModelVersionManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LearningConfig:
    """Configuration for online learning pipeline."""
    model_path: str
    watch_directory: str
    output_directory: str
    batch_size: int = 4
    learning_rate: float = 1e-5
    validation_threshold: float = 0.8
    min_recipes_for_training: int = 10
    max_recipes_per_batch: int = 100
    training_interval_hours: int = 6
    validation_split: float = 0.2
    enable_versioning: bool = True
    max_versions: int = 10

class RecipeValidator:
    """Validates new recipes before training."""
    
    def __init__(self, tokenizer: GPT2Tokenizer):
        self.tokenizer = tokenizer
        
    def validate_recipe(self, recipe: Dict) -> Tuple[bool, List[str]]:
        """Validate a single recipe and return validation result with reasons."""
        issues = []
        
        # Required fields check
        required_fields = ['title', 'ingredients', 'instructions']
        for field in required_fields:
            if field not in recipe or not recipe[field]:
                issues.append(f"Missing or empty {field}")
                
        # Content quality checks
        if 'ingredients' in recipe:
            ingredients = recipe['ingredients']
            if isinstance(ingredients, str):
                ingredients = ingredients.split('|')
            if len(ingredients) < 2:
                issues.append("Too few ingredients (minimum 2)")
            if len(ingredients) > 50:
                issues.append("Too many ingredients (maximum 50)")
                
        if 'instructions' in recipe:
            instructions = recipe['instructions']
            if len(instructions) < 20:
                issues.append("Instructions too short (minimum 20 characters)")
            if len(instructions) > 2000:
                issues.append("Instructions too long (maximum 2000 characters)")
                
        # Tokenization check
        if not issues:  # Only check if basic validation passes
            try:
                formatted_text = self._format_recipe(recipe)
                tokens = self.tokenizer.encode(formatted_text)
                if len(tokens) > 512:
                    issues.append("Recipe too long when tokenized (>512 tokens)")
            except Exception as e:
                issues.append(f"Tokenization error: {str(e)}")
                
        return len(issues) == 0, issues
        
    def _format_recipe(self, recipe: Dict) -> str:
        """Format recipe for tokenization test."""
        ingredients_text = " | ".join(recipe['ingredients']) if isinstance(recipe['ingredients'], list) else recipe['ingredients']
        return f"<TITLE>{recipe['title']}</TITLE><INGREDIENTS>{ingredients_text}</INGREDIENTS><INSTRUCTIONS>{recipe['instructions']}</INSTRUCTIONS>"

class OnlineLearningPipeline:
    """Main pipeline for online learning from new recipes."""
    
    def __init__(self, config: LearningConfig):
        self.config = config
        self.model_path = Path(config.model_path)
        self.watch_dir = Path(config.watch_directory)
        self.output_dir = Path(config.output_directory)
        
        # Create directories
        self.watch_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "processed").mkdir(exist_ok=True)
        (self.output_dir / "failed").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Initialize components
        self.tokenizer = None
        self.validator = None
        self.trainer = None
        self.version_manager = None
        
        # Runtime state
        self.running = False
        self.recipe_queue = queue.Queue()
        self.stats = {
            'total_processed': 0,
            'total_trained': 0,
            'validation_failures': 0,
            'training_sessions': 0,
            'last_training': None,
            'current_batch_size': 0
        }
        
        self._initialize_components()
        
    def _initialize_components(self):
        """Initialize model and related components."""
        logger.info(f"Initializing components with model: {self.model_path}")
        
        # Load tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(self.model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize validator
        self.validator = RecipeValidator(self.tokenizer)
        
        # Initialize trainer
        training_config = {
            'incremental_epochs': 2,
            'incremental_lr': self.config.learning_rate,
            'batch_size': self.config.batch_size,
            'max_length': 512,
            'warmup_steps': 50,
            'save_steps': 100,
        }
        self.trainer = IncrementalTrainer(str(self.model_path), training_config)
        self.trainer.load_existing_model()
        
        # Initialize version manager if enabled
        if self.config.enable_versioning:
            self.version_manager = ModelVersionManager(str(self.model_path.parent))
            
    def watch_for_new_recipes(self):
        """Watch directory for new recipe files."""
        logger.info(f"Watching directory: {self.watch_dir}")
        
        processed_files = set()
        
        while self.running:
            try:
                # Look for new CSV files
                for csv_file in self.watch_dir.glob("*.csv"):
                    if csv_file.name not in processed_files:
                        logger.info(f"Found new recipe file: {csv_file}")
                        self._process_new_file(csv_file)
                        processed_files.add(csv_file.name)
                        
                # Check if we have enough recipes to start training
                if self.recipe_queue.qsize() >= self.config.min_recipes_for_training:
                    self._trigger_training()
                    
                # Check for scheduled training
                self._check_scheduled_training()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in file watching: {e}")
                time.sleep(60)  # Wait longer on error
                
    def _process_new_file(self, csv_file: Path):
        """Process a new CSV file with recipes."""
        try:
            recipes = load_new_recipes_from_csv(str(csv_file))
            logger.info(f"Loaded {len(recipes)} recipes from {csv_file}")
            
            # Validate and queue recipes
            valid_count = 0
            for recipe in recipes:
                is_valid, issues = self.validator.validate_recipe(recipe)
                if is_valid:
                    self.recipe_queue.put(recipe)
                    valid_count += 1
                else:
                    logger.warning(f"Invalid recipe '{recipe.get('title', 'Unknown')}': {', '.join(issues)}")
                    self.stats['validation_failures'] += 1
                    
            logger.info(f"Queued {valid_count} valid recipes from {csv_file}")
            self.stats['total_processed'] += len(recipes)
            self.stats['current_batch_size'] = self.recipe_queue.qsize()
            
            # Move processed file
            processed_path = self.output_dir / "processed" / csv_file.name
            csv_file.rename(processed_path)
            
        except Exception as e:
            logger.error(f"Error processing file {csv_file}: {e}")
            # Move to failed directory
            failed_path = self.output_dir / "failed" / csv_file.name
            csv_file.rename(failed_path)
            
    def _trigger_training(self):
        """Trigger training with queued recipes."""
        if self.recipe_queue.empty():
            return
            
        # Collect recipes from queue
        recipes = []
        max_recipes = min(self.config.max_recipes_per_batch, self.recipe_queue.qsize())
        
        for _ in range(max_recipes):
            if not self.recipe_queue.empty():
                recipes.append(self.recipe_queue.get())
                
        if not recipes:
            return
            
        logger.info(f"Starting training with {len(recipes)} recipes")
        
        try:
            # Create version backup if enabled
            if self.config.enable_versioning:
                pre_training_version = self.version_manager.create_version(
                    str(self.model_path),
                    description=f"Pre-training checkpoint before batch of {len(recipes)} recipes",
                    metrics=self.stats.copy()
                )
                logger.info(f"Created pre-training version: {pre_training_version}")
                
            # Perform training
            training_output = self.output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.trainer.train_incremental(recipes, str(training_output))
            
            # Update model path to new training output
            if training_output.exists():
                # Copy trained model back to main location
                import shutil
                for file_path in training_output.glob("*"):
                    if file_path.is_file():
                        shutil.copy2(file_path, self.model_path / file_path.name)
                        
                # Reload trainer with updated model
                self.trainer.load_existing_model()
                
            # Create post-training version if enabled
            if self.config.enable_versioning:
                post_training_version = self.version_manager.create_version(
                    str(self.model_path),
                    description=f"After training on {len(recipes)} recipes",
                    metrics={
                        **self.stats,
                        'recipes_in_batch': len(recipes),
                        'training_completed_at': datetime.now().isoformat()
                    }
                )
                logger.info(f"Created post-training version: {post_training_version}")
                
                # Cleanup old versions
                self.version_manager.cleanup_old_versions(self.config.max_versions)
                
            # Update stats
            self.stats['total_trained'] += len(recipes)
            self.stats['training_sessions'] += 1
            self.stats['last_training'] = datetime.now().isoformat()
            self.stats['current_batch_size'] = self.recipe_queue.qsize()
            
            # Save stats
            self._save_stats()
            
            logger.info(f"Training completed successfully. Trained on {len(recipes)} recipes.")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            # Put recipes back in queue for retry
            for recipe in recipes:
                self.recipe_queue.put(recipe)
                
    def _check_scheduled_training(self):
        """Check if scheduled training should be triggered."""
        if not self.stats['last_training']:
            return
            
        last_training = datetime.fromisoformat(self.stats['last_training'])
        time_since_training = datetime.now() - last_training
        
        if (time_since_training.total_seconds() / 3600 >= self.config.training_interval_hours and 
            not self.recipe_queue.empty()):
            logger.info("Triggering scheduled training")
            self._trigger_training()
            
    def _save_stats(self):
        """Save current statistics."""
        stats_file = self.output_dir / "logs" / "pipeline_stats.json"
        with open(stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
            
    def get_status(self) -> Dict:
        """Get current pipeline status."""
        return {
            'running': self.running,
            'queued_recipes': self.recipe_queue.qsize(),
            'stats': self.stats.copy(),
            'config': {
                'model_path': str(self.model_path),
                'watch_directory': str(self.watch_dir),
                'min_recipes_for_training': self.config.min_recipes_for_training,
                'training_interval_hours': self.config.training_interval_hours,
            }
        }
        
    def start(self):
        """Start the online learning pipeline."""
        if self.running:
            logger.warning("Pipeline is already running")
            return
            
        self.running = True
        logger.info("Starting online learning pipeline")
        
        # Start file watching in a separate thread
        self.watch_thread = threading.Thread(target=self.watch_for_new_recipes, daemon=True)
        self.watch_thread.start()
        
        logger.info("Online learning pipeline started")
        
    def stop(self):
        """Stop the online learning pipeline."""
        if not self.running:
            logger.warning("Pipeline is not running")
            return
            
        self.running = False
        logger.info("Stopping online learning pipeline")
        
        if hasattr(self, 'watch_thread'):
            self.watch_thread.join(timeout=10)
            
        self._save_stats()
        logger.info("Online learning pipeline stopped")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Online learning pipeline for recipe model")
    parser.add_argument("--model-path", required=True, help="Path to base model")
    parser.add_argument("--watch-dir", required=True, help="Directory to watch for new recipe files")
    parser.add_argument("--output-dir", required=True, help="Output directory for training results")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--min-recipes", type=int, default=10, help="Minimum recipes to trigger training")
    parser.add_argument("--max-recipes", type=int, default=100, help="Maximum recipes per training batch")
    parser.add_argument("--training-interval", type=int, default=6, help="Hours between scheduled training")
    parser.add_argument("--no-versioning", action="store_true", help="Disable model versioning")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon (background process)")
    
    args = parser.parse_args()
    
    # Create configuration
    config = LearningConfig(
        model_path=args.model_path,
        watch_directory=args.watch_dir,
        output_directory=args.output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        min_recipes_for_training=args.min_recipes,
        max_recipes_per_batch=args.max_recipes,
        training_interval_hours=args.training_interval,
        enable_versioning=not args.no_versioning
    )
    
    # Initialize pipeline
    pipeline = OnlineLearningPipeline(config)
    
    if args.daemon:
        # Run as daemon
        pipeline.start()
        try:
            while True:
                time.sleep(60)
                status = pipeline.get_status()
                logger.info(f"Pipeline status - Queued: {status['queued_recipes']}, "
                           f"Total trained: {status['stats']['total_trained']}")
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            pipeline.stop()
    else:
        # Interactive mode
        pipeline.start()
        
        try:
            print("Online learning pipeline started. Commands:")
            print("  status - Show pipeline status")
            print("  stop   - Stop the pipeline")
            print("  quit   - Stop and exit")
            
            while pipeline.running:
                try:
                    command = input("> ").strip().lower()
                    
                    if command == "status":
                        status = pipeline.get_status()
                        print(f"Running: {status['running']}")
                        print(f"Queued recipes: {status['queued_recipes']}")
                        print(f"Total processed: {status['stats']['total_processed']}")
                        print(f"Total trained: {status['stats']['total_trained']}")
                        print(f"Training sessions: {status['stats']['training_sessions']}")
                        print(f"Last training: {status['stats']['last_training']}")
                        
                    elif command in ["stop", "quit"]:
                        break
                        
                    else:
                        print("Unknown command. Available: status, stop, quit")
                        
                except EOFError:
                    break
                    
        except KeyboardInterrupt:
            logger.info("Received interrupt signal")
        finally:
            pipeline.stop()


if __name__ == "__main__":
    main()