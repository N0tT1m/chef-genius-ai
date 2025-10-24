#!/usr/bin/env python3
"""
Convert existing FLAN-T5 training data to Mistral chat template format
"""

import json
import os
from pathlib import Path
from transformers import AutoTokenizer

def convert_dataset_to_mistral_format(input_file: str, output_file: str, tokenizer):
    """Convert FLAN-T5 format to Mistral chat template format."""
    
    print(f"Converting {input_file} to Mistral format...")
    
    converted_data = []
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for item in data:
            try:
                # Extract original data
                input_data = item.get('input_data', {})
                output_data = item.get('output_data', {})
                
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
                ]\n                
                # Apply chat template
                formatted_text = tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                converted_data.append({
                    'text': formatted_text,
                    'original_input': input_data,
                    'original_output': output_data
                })
                
            except Exception as e:
                print(f"Error converting item: {e}")
                continue
        
        # Save converted data
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(converted_data, f, indent=2, ensure_ascii=False)
        
        print(f"Converted {len(converted_data)} items to {output_file}")
        
    except Exception as e:
        print(f"Error processing {input_file}: {e}")

def main():
    """Main conversion function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert FLAN-T5 data to Mistral format")
    parser.add_argument("--data-dir", default="cli/data/datasets", help="Data directory")
    parser.add_argument("--output-dir", default="cli/data/mistral_datasets", help="Output directory")
    
    args = parser.parse_args()
    
    print("🔄 CONVERTING DATA TO MISTRAL FORMAT")
    print("="*40)
    
    # Load Mistral tokenizer
    print("Loading Mistral tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all training.json files
    data_path = Path(args.data_dir)
    training_files = list(data_path.glob("**/training.json"))
    
    print(f"Found {len(training_files)} training files")
    
    for file_path in training_files:
        dataset_name = file_path.parent.name
        output_file = output_dir / f"{dataset_name}_mistral.json"
        
        convert_dataset_to_mistral_format(str(file_path), str(output_file), tokenizer)
    
    print(f"\n✅ Conversion completed!")
    print(f"📁 Converted data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()