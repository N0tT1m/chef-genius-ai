#!/usr/bin/env python3
"""
Convert cleaned recipe JSONL files to FLAN-T5 training format
Processes the output from recipe_cleaner and creates training-ready data
"""

import json
import sys
from pathlib import Path
from tqdm import tqdm
import argparse


def calculate_quality_score(recipe):
    """Calculate quality score for a recipe."""
    score = 0.0

    # Has title
    if recipe.get('title'):
        score += 0.3

    # Has ingredients
    ingredients = recipe.get('ingredients', [])
    if isinstance(ingredients, str):
        ingredients = [i.strip() for i in ingredients.split(',') if i.strip()]

    if ingredients and len(ingredients) > 0:
        score += 0.35
        # Bonus for good ingredient count
        if len(ingredients) >= 3:
            score += 0.1

    # Has instructions (can be missing in some datasets)
    instructions = recipe.get('instructions', [])
    if isinstance(instructions, str):
        instructions = [i.strip() for i in instructions.split('.') if i.strip() and len(i.strip()) > 10]

    if instructions and len(instructions) > 0:
        score += 0.25

    return min(score, 1.0)


def convert_to_flan_t5_format(recipe):
    """Convert a cleaned recipe to FLAN-T5 training format."""

    # Extract fields
    title = recipe.get('title', '').strip()

    # Handle ingredients (can be array or string)
    ingredients = recipe.get('ingredients', [])
    if isinstance(ingredients, str):
        try:
            # Try parsing as JSON array
            ingredients = json.loads(ingredients)
        except:
            # Split by common delimiters
            ingredients = [i.strip() for i in ingredients.replace('|', ',').split(',') if i.strip()]

    # Handle instructions
    instructions = recipe.get('instructions', [])
    if isinstance(instructions, str):
        try:
            # Try parsing as JSON array
            instructions = json.loads(instructions)
        except:
            # Split by sentences
            instructions = [i.strip() for i in instructions.split('.') if i.strip() and len(i.strip()) > 10]

    # Skip if missing critical fields
    if not title or not ingredients:
        return None

    # Create FLAN-T5 instruction format
    # Input: instruction-style prompt
    input_text = f"Create a detailed recipe for {title} with step-by-step instructions."

    # Output: structured recipe
    output_parts = [f"# {title}", ""]

    if ingredients:
        output_parts.append("## Ingredients")
        for ing in ingredients:
            if ing.strip():
                output_parts.append(f"- {ing.strip()}")
        output_parts.append("")

    if instructions:
        output_parts.append("## Instructions")
        for i, inst in enumerate(instructions, 1):
            if inst.strip():
                output_parts.append(f"{i}. {inst.strip()}")

    output_text = "\n".join(output_parts)

    # Calculate quality score
    quality_score = calculate_quality_score(recipe)

    return {
        "input": input_text,
        "output": output_text,
        "quality_score": quality_score,
        "metadata": {
            "title": title,
            "ingredient_count": len(ingredients),
            "instruction_count": len(instructions)
        }
    }


def process_jsonl_file(input_file, output_file, min_quality=0.5):
    """Process a cleaned JSONL file and convert to FLAN-T5 format."""

    print(f"📂 Processing: {input_file}")
    print(f"💾 Output: {output_file}")
    print(f"🎯 Min quality: {min_quality}")

    # Count total lines
    with open(input_file, 'r', encoding='utf-8') as f:
        total_lines = sum(1 for _ in f)

    print(f"📊 Total recipes: {total_lines:,}")

    processed = 0
    kept = 0
    filtered_low_quality = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in tqdm(infile, total=total_lines, desc="Converting"):
            line = line.strip()
            if not line:
                continue

            try:
                recipe = json.loads(line)
                processed += 1

                # Convert to FLAN-T5 format
                flan_t5_recipe = convert_to_flan_t5_format(recipe)

                if flan_t5_recipe and flan_t5_recipe['quality_score'] >= min_quality:
                    outfile.write(json.dumps(flan_t5_recipe) + '\n')
                    kept += 1
                else:
                    filtered_low_quality += 1

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON error: {e}")
                continue
            except Exception as e:
                print(f"⚠️  Error processing recipe: {e}")
                continue

    print(f"\n✅ Conversion complete!")
    print(f"   Processed: {processed:,}")
    print(f"   Kept: {kept:,} ({kept/processed*100:.1f}%)")
    print(f"   Filtered: {filtered_low_quality:,} ({filtered_low_quality/processed*100:.1f}%)")

    return kept


def main():
    parser = argparse.ArgumentParser(description='Convert cleaned JSONL to FLAN-T5 format')
    parser.add_argument('--input', '-i', required=True, help='Input cleaned JSONL file')
    parser.add_argument('--output', '-o', required=True, help='Output FLAN-T5 JSONL file')
    parser.add_argument('--min-quality', type=float, default=0.5, help='Minimum quality score (default: 0.5)')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}")
        sys.exit(1)

    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Process the file
    process_jsonl_file(input_path, output_path, args.min_quality)


if __name__ == "__main__":
    main()
