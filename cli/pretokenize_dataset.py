#!/usr/bin/env python3
"""
Pre-tokenize Recipe Dataset for Faster Training
Tokenizes the entire dataset once and caches it, eliminating runtime tokenization bottleneck.
"""

import argparse
import json
import logging
import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import T5Tokenizer
from tqdm import tqdm

# Import from the main training script
from train_recipe_model import RecipeGenerationModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetPreTokenizer:
    """Pre-tokenizes datasets for faster training."""
    
    def __init__(self, model_type: str = "t5", pretrained_model: str = "google/flan-t5-xl", max_length: int = 512):
        self.model_type = model_type
        self.pretrained_model = pretrained_model
        self.max_length = max_length
        self.tokenizer = None
        
    def initialize_tokenizer(self):
        """Initialize the tokenizer."""
        logger.info(f"Initializing tokenizer for {self.pretrained_model}")
        
        if self.model_type == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained(self.pretrained_model)
        else:
            from transformers import GPT2Tokenizer
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"Tokenizer vocab size: {len(self.tokenizer)}")
    
    def pretokenize_dataset(self, recipes_data: List[Dict], output_path: str, dataset_name: str = "recipes"):
        """Pre-tokenize and cache the entire dataset."""
        logger.info(f"Pre-tokenizing {len(recipes_data)} recipes...")
        
        tokenized_data = []
        failed_count = 0
        
        # Use the same recipe formatting as the main training script
        from train_recipe_model import RecipeDataset
        temp_dataset = RecipeDataset(
            recipes_data, 
            self.tokenizer, 
            max_length=self.max_length,
            model_type=self.model_type,
            multi_task=False
        )
        
        for i in tqdm(range(len(temp_dataset)), desc="Tokenizing recipes"):
            try:
                tokenized_sample = temp_dataset[i]
                
                # Convert tensors to lists for JSON serialization
                serializable_sample = {}
                for key, value in tokenized_sample.items():
                    if torch.is_tensor(value):
                        serializable_sample[key] = value.tolist()
                    else:
                        serializable_sample[key] = value
                
                tokenized_data.append(serializable_sample)
                
            except Exception as e:
                logger.debug(f"Failed to tokenize recipe {i}: {e}")
                failed_count += 1
                continue
        
        logger.info(f"Successfully tokenized {len(tokenized_data)} recipes, failed: {failed_count}")
        
        # Save tokenized data
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as pickle for faster loading
        pickle_path = output_path.replace('.json', '.pkl')
        with open(pickle_path, 'wb') as f:
            pickle.dump(tokenized_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Also save metadata
        metadata = {
            'dataset_name': dataset_name,
            'model_type': self.model_type,
            'pretrained_model': self.pretrained_model,
            'max_length': self.max_length,
            'num_samples': len(tokenized_data),
            'tokenizer_vocab_size': len(self.tokenizer),
            'failed_count': failed_count
        }
        
        metadata_path = output_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Pre-tokenized dataset saved to: {pickle_path}")
        logger.info(f"Metadata saved to: {metadata_path}")
        
        return pickle_path, metadata_path

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize recipe dataset for faster training")
    parser.add_argument("--datasets", nargs="*", help="Specific datasets to tokenize (uses all available if not specified)")
    parser.add_argument("--output-dir", default="data/tokenized", help="Output directory for tokenized data")
    parser.add_argument("--model-type", choices=["gpt2", "t5"], default="t5", help="Model architecture")
    parser.add_argument("--pretrained-model", default="google/flan-t5-xl", help="Pretrained model to use for tokenizer")
    parser.add_argument("--max-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--max-samples", type=int, help="Maximum number of recipes to tokenize (for testing)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize pre-tokenizer
    pretokenizer = DatasetPreTokenizer(
        model_type=args.model_type,
        pretrained_model=args.pretrained_model,
        max_length=args.max_length
    )
    
    pretokenizer.initialize_tokenizer()
    
    # Initialize the main model to access dataset loading functionality
    model_config = {
        'model_type': args.model_type,
        'pretrained_model': args.pretrained_model,
        'max_sequence_length': args.max_length,
        'filter_quality': True,
        'use_augmentation': False
    }
    
    trainer = RecipeGenerationModel(model_config)
    trainer.max_samples = args.max_samples
    trainer.tokenizer = pretokenizer.tokenizer  # Use the same tokenizer
    
    # Load dataset
    logger.info("Loading dataset...")
    recipes_data = trainer._load_local_datasets(args.datasets)
    
    if not recipes_data:
        logger.error("No recipes loaded. Check your dataset paths.")
        return
    
    # Apply quality filtering
    recipes_data = trainer._filter_recipe_quality(recipes_data)
    
    # Limit samples if specified
    if args.max_samples and len(recipes_data) > args.max_samples:
        logger.info(f"Limiting to {args.max_samples} samples for testing")
        recipes_data = recipes_data[:args.max_samples]
    
    logger.info(f"Processing {len(recipes_data)} recipes")
    
    # Create output filename
    dataset_names = "_".join(args.datasets) if args.datasets else "all_datasets"
    output_filename = f"{dataset_names}_{args.model_type}_{args.max_length}tokens.json"
    output_path = os.path.join(args.output_dir, output_filename)
    
    # Pre-tokenize the dataset
    pickle_path, metadata_path = pretokenizer.pretokenize_dataset(
        recipes_data, 
        output_path, 
        dataset_names
    )
    
    logger.info("Pre-tokenization completed successfully!")
    logger.info(f"To use this pre-tokenized dataset, modify your training script to load from: {pickle_path}")
    
    # Print memory usage estimate
    import sys
    file_size_mb = os.path.getsize(pickle_path) / (1024 * 1024)
    logger.info(f"Pre-tokenized dataset size: {file_size_mb:.1f} MB")
    
    # Create a simple usage example
    usage_example = f"""
# Usage in training script:
import pickle

# Load pre-tokenized data
with open('{pickle_path}', 'rb') as f:
    tokenized_data = pickle.load(f)

# Convert back to tensors and use directly
for sample in tokenized_data:
    for key, value in sample.items():
        if isinstance(value, list):
            sample[key] = torch.tensor(value)
"""
    
    with open(os.path.join(args.output_dir, "usage_example.py"), 'w') as f:
        f.write(usage_example)

if __name__ == "__main__":
    main()