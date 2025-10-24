#!/usr/bin/env python3
"""
Test recipe generation from trained model
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_recipe_generation():
    """Test the trained model generation."""
    model_path = "../models/recipe_generation_small"
    
    # Load tokenizer and model
    logger.info(f"Loading model from {model_path}")
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    
    # Device setup
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    model.to(device)
    model.eval()
    
    # Test prompts
    test_prompts = [
        "<TITLE>",
        "<TITLE>Chocolate Chip Cookies",
        "<TITLE>Pasta Carbonara<CUISINE>Italian",
        "<TITLE>Chicken Curry<CUISINE>Indian<TIME>30 minutes"
    ]
    
    logger.info("Generating recipes...")
    
    for i, prompt in enumerate(test_prompts, 1):
        logger.info(f"\n--- Test {i} ---")
        logger.info(f"Prompt: {prompt}")
        
        with torch.no_grad():
            # Encode prompt
            input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
            
            # Generate
            output = model.generate(
                input_ids,
                max_length=300,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                no_repeat_ngram_size=2
            )
            
            # Decode
            generated_text = tokenizer.decode(output[0], skip_special_tokens=False)
            logger.info(f"Generated: {generated_text}")

if __name__ == "__main__":
    test_recipe_generation()