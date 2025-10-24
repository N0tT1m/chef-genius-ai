#!/usr/bin/env python3
"""
CSV to Training Data Parser
Parses various CSV recipe formats and converts them to the standard training JSON format
"""

import argparse
import csv
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CSVRecipeParser:
    """Parser for converting CSV recipe files to training format."""
    
    def __init__(self):
        self.supported_formats = {
            'recipe_dataset_simple': self._parse_recipe_dataset_simple,
            'recipe_box': self._parse_recipe_box,
            'indian_recipe_api': self._parse_indian_recipe_api,
            'recipe_nlg': self._parse_recipe_nlg,
            'indian_food_analysis': self._parse_indian_food_analysis,
            'allrecipes_ingredients': self._parse_allrecipes_ingredients,
        }
    
    def parse_csv_to_training_format(self, csv_path: str, format_type: str = None, output_path: str = None) -> List[Dict]:
        """Parse a CSV file and convert to training format."""
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Auto-detect format if not specified
        if format_type is None:
            format_type = self._detect_format(csv_path)
        
        if format_type not in self.supported_formats:
            raise ValueError(f"Unsupported format: {format_type}. Supported formats: {list(self.supported_formats.keys())}")
        
        logger.info(f"Parsing {csv_path} as format: {format_type}")
        
        # Parse the CSV
        parser_func = self.supported_formats[format_type]
        recipes_data = parser_func(csv_path)
        
        logger.info(f"Parsed {len(recipes_data)} recipes from {csv_path}")
        
        # Save to output file if specified
        if output_path:
            self._save_training_data(recipes_data, output_path)
        
        return recipes_data
    
    def _detect_format(self, csv_path: str) -> str:
        """Auto-detect the CSV format based on filename and headers."""
        filename = os.path.basename(csv_path).lower()
        
        # Check filename patterns
        if '13k-recipes' in filename or 'recipe_dataset_simple' in filename:
            return 'recipe_dataset_simple'
        elif 'recipe_box' in filename:
            return 'recipe_box'
        elif 'indian_recipe_api' in filename or 'indianfooddataset' in filename:
            return 'indian_recipe_api'
        elif 'recipe_nlg' in filename or 'recipenlg' in filename:
            return 'recipe_nlg'
        elif 'indian_food_analysis' in filename:
            return 'indian_food_analysis'
        elif 'ingredients' in filename and 'allrecipes' in filename:
            return 'allrecipes_ingredients'
        
        # Try to detect from headers
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                headers = next(reader)
                headers_lower = [h.lower() for h in headers]
                
                if 'title' in headers_lower and 'ingredients' in headers_lower and 'instructions' in headers_lower:
                    if 'image_name' in headers_lower:
                        return 'recipe_dataset_simple'
                    elif 'directions' in headers_lower:
                        return 'recipe_box'
                elif 'recipename' in headers_lower and 'cuisine' in headers_lower:
                    return 'indian_recipe_api'
                elif 'recipe' in headers_lower and 'ner' in headers_lower:
                    return 'recipe_nlg'
                
        except Exception as e:
            logger.warning(f"Could not auto-detect format: {e}")
        
        logger.warning("Could not auto-detect format, defaulting to 'recipe_dataset_simple'")
        return 'recipe_dataset_simple'
    
    def _parse_recipe_dataset_simple(self, csv_path: str) -> List[Dict]:
        """Parse 13k-recipes.csv format."""
        recipes = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    recipe = {
                        "input_data": {},
                        "output_data": {}
                    }
                    
                    # Title
                    title = row.get('Title', '').strip()
                    if title:
                        recipe["output_data"]["title"] = title
                    
                    # Ingredients
                    ingredients_raw = row.get('Ingredients', '') or row.get('Cleaned_Ingredients', '')
                    if ingredients_raw:
                        ingredients = self._parse_ingredients_list(ingredients_raw)
                        if ingredients:
                            recipe["input_data"]["ingredients"] = ingredients
                    
                    # Instructions
                    instructions = row.get('Instructions', '').strip()
                    if instructions:
                        recipe["output_data"]["instructions"] = self._parse_instructions(instructions)
                    
                    # Only add if has title and ingredients or instructions
                    if (recipe["output_data"].get("title") and 
                        (recipe["input_data"].get("ingredients") or recipe["output_data"].get("instructions"))):
                        recipes.append(recipe)
                        
                except Exception as e:
                    logger.debug(f"Error parsing recipe: {e}")
                    continue
        
        return recipes
    
    def _parse_recipe_box(self, csv_path: str) -> List[Dict]:
        """Parse recipe_box format with ingredient columns."""
        recipes = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    recipe = {
                        "input_data": {},
                        "output_data": {}
                    }
                    
                    # Title
                    title = row.get('Title', '').strip()
                    if title:
                        recipe["output_data"]["title"] = title
                    
                    # Instructions/Directions
                    directions = row.get('Directions', '').strip()
                    if directions:
                        recipe["output_data"]["instructions"] = self._parse_instructions(directions)
                    
                    # Parse ingredients from multiple columns
                    ingredients = []
                    for i in range(1, 20):  # Up to 19 ingredients
                        qty_col = f'Quantity{i:02d}' if i > 1 else 'Quantity'
                        unit_col = f'Unit{i:02d}' if i > 1 else 'Unit01'
                        ing_col = f'Ingredient{i:02d}' if i > 1 else 'Ingredient01'
                        
                        qty = row.get(qty_col, '').strip()
                        unit = row.get(unit_col, '').strip()
                        ingredient = row.get(ing_col, '').strip()
                        
                        if ingredient and ingredient.lower() not in ['', 'nan', 'none']:
                            ingredient_text = f"{qty} {unit} {ingredient}".strip()
                            ingredients.append(ingredient_text)
                    
                    if ingredients:
                        recipe["input_data"]["ingredients"] = ingredients
                    
                    # Category as cuisine
                    category = row.get('Category', '').strip()
                    if category:
                        recipe["input_data"]["cuisine"] = category
                    
                    # Only add if has title and ingredients or instructions
                    if (recipe["output_data"].get("title") and 
                        (recipe["input_data"].get("ingredients") or recipe["output_data"].get("instructions"))):
                        recipes.append(recipe)
                        
                except Exception as e:
                    logger.debug(f"Error parsing recipe: {e}")
                    continue
        
        return recipes
    
    def _parse_indian_recipe_api(self, csv_path: str) -> List[Dict]:
        """Parse Indian recipe API format."""
        recipes = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    recipe = {
                        "input_data": {},
                        "output_data": {}
                    }
                    
                    # Recipe name
                    recipe_name = row.get('RecipeName', '') or row.get('TranslatedRecipeName', '')
                    if recipe_name:
                        recipe["output_data"]["title"] = recipe_name.strip()
                    
                    # Ingredients
                    ingredients_raw = row.get('Ingredients', '') or row.get('TranslatedIngredients', '')
                    if ingredients_raw:
                        ingredients = self._parse_ingredients_list(ingredients_raw)
                        if ingredients:
                            recipe["input_data"]["ingredients"] = ingredients
                    
                    # Instructions
                    instructions_raw = row.get('Instructions', '') or row.get('TranslatedInstructions', '')
                    if instructions_raw:
                        recipe["output_data"]["instructions"] = self._parse_instructions(instructions_raw)
                    
                    # Additional metadata
                    cuisine = row.get('Cuisine', '').strip()
                    if cuisine:
                        recipe["input_data"]["cuisine"] = cuisine
                    
                    course = row.get('Course', '').strip()
                    if course:
                        recipe["input_data"]["course"] = course
                    
                    prep_time = row.get('PrepTimeInMins', '').strip()
                    cook_time = row.get('CookTimeInMins', '').strip()
                    total_time = row.get('TotalTimeInMins', '').strip()
                    
                    if total_time:
                        recipe["input_data"]["cooking_time"] = f"{total_time} minutes"
                    elif cook_time:
                        recipe["input_data"]["cooking_time"] = f"{cook_time} minutes"
                    
                    servings = row.get('Servings', '').strip()
                    if servings:
                        recipe["input_data"]["servings"] = servings
                    
                    # Only add if has title and ingredients or instructions
                    if (recipe["output_data"].get("title") and 
                        (recipe["input_data"].get("ingredients") or recipe["output_data"].get("instructions"))):
                        recipes.append(recipe)
                        
                except Exception as e:
                    logger.debug(f"Error parsing recipe: {e}")
                    continue
        
        return recipes
    
    def _parse_recipe_nlg(self, csv_path: str) -> List[Dict]:
        """Parse RecipeNLG dataset format."""
        recipes = []
        
        # This file is usually very large, so we'll read it in chunks
        chunk_size = 1000
        processed_count = 0
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    recipe = {
                        "input_data": {},
                        "output_data": {}
                    }
                    
                    # Title
                    title = row.get('title', '').strip()
                    if title:
                        recipe["output_data"]["title"] = title
                    
                    # Ingredients
                    ingredients_raw = row.get('ingredients', '')
                    if ingredients_raw:
                        ingredients = self._parse_ingredients_list(ingredients_raw)
                        if ingredients:
                            recipe["input_data"]["ingredients"] = ingredients
                    
                    # Directions
                    directions = row.get('directions', '').strip()
                    if directions:
                        recipe["output_data"]["instructions"] = self._parse_instructions(directions)
                    
                    # Link (optional)
                    link = row.get('link', '').strip()
                    if link:
                        recipe["input_data"]["source_url"] = link
                    
                    # Only add if has title and ingredients or instructions
                    if (recipe["output_data"].get("title") and 
                        (recipe["input_data"].get("ingredients") or recipe["output_data"].get("instructions"))):
                        recipes.append(recipe)
                        processed_count += 1
                        
                        # Log progress for large files
                        if processed_count % chunk_size == 0:
                            logger.info(f"Processed {processed_count} recipes...")
                        
                except Exception as e:
                    logger.debug(f"Error parsing recipe: {e}")
                    continue
        
        return recipes
    
    def _parse_indian_food_analysis(self, csv_path: str) -> List[Dict]:
        """Parse Indian food analysis format."""
        recipes = []
        
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                try:
                    recipe = {
                        "input_data": {},
                        "output_data": {}
                    }
                    
                    # Use available columns (this format may vary)
                    for col_name, col_value in row.items():
                        col_name_lower = col_name.lower().strip()
                        col_value = col_value.strip() if col_value else ''
                        
                        if not col_value:
                            continue
                        
                        if 'name' in col_name_lower or 'title' in col_name_lower:
                            recipe["output_data"]["title"] = col_value
                        elif 'ingredient' in col_name_lower:
                            ingredients = self._parse_ingredients_list(col_value)
                            if ingredients:
                                recipe["input_data"]["ingredients"] = ingredients
                        elif 'instruction' in col_name_lower or 'direction' in col_name_lower or 'method' in col_name_lower:
                            recipe["output_data"]["instructions"] = self._parse_instructions(col_value)
                        elif 'cuisine' in col_name_lower:
                            recipe["input_data"]["cuisine"] = col_value
                        elif 'time' in col_name_lower:
                            recipe["input_data"]["cooking_time"] = col_value
                        elif 'serving' in col_name_lower:
                            recipe["input_data"]["servings"] = col_value
                    
                    # Only add if has title and ingredients or instructions
                    if (recipe["output_data"].get("title") and 
                        (recipe["input_data"].get("ingredients") or recipe["output_data"].get("instructions"))):
                        recipes.append(recipe)
                        
                except Exception as e:
                    logger.debug(f"Error parsing recipe: {e}")
                    continue
        
        return recipes
    
    def _parse_allrecipes_ingredients(self, csv_path: str) -> List[Dict]:
        """Parse allrecipes ingredients format."""
        recipes = []
        
        # This format might just be ingredients lists, so we'll create basic recipes
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                try:
                    recipe = {
                        "input_data": {},
                        "output_data": {}
                    }
                    
                    # Create a generic title
                    recipe["output_data"]["title"] = f"Recipe {i+1}"
                    
                    # Look for ingredient columns
                    ingredients = []
                    for col_name, col_value in row.items():
                        if col_value and col_value.strip():
                            ingredients.append(col_value.strip())
                    
                    if ingredients:
                        recipe["input_data"]["ingredients"] = ingredients
                        recipes.append(recipe)
                        
                except Exception as e:
                    logger.debug(f"Error parsing recipe: {e}")
                    continue
        
        return recipes
    
    def _parse_ingredients_list(self, ingredients_text: str) -> List[str]:
        """Parse ingredients from various text formats."""
        if not ingredients_text or ingredients_text.strip() == '':
            return []
        
        # Remove extra quotes and brackets
        ingredients_text = ingredients_text.strip().strip('"\'')
        
        # Handle list-like strings
        if ingredients_text.startswith('[') and ingredients_text.endswith(']'):
            try:
                # Try to parse as Python list
                ingredients_list = eval(ingredients_text)
                if isinstance(ingredients_list, list):
                    return [str(ing).strip().strip('"\'') for ing in ingredients_list if str(ing).strip()]
            except:
                # If eval fails, parse manually
                ingredients_text = ingredients_text[1:-1]  # Remove brackets
        
        # Split by common delimiters
        separators = [',', '|', ';', '\n']
        ingredients = [ingredients_text]
        
        for sep in separators:
            new_ingredients = []
            for ing in ingredients:
                new_ingredients.extend([part.strip() for part in ing.split(sep) if part.strip()])
            ingredients = new_ingredients
        
        # Clean up ingredients
        cleaned_ingredients = []
        for ing in ingredients:
            ing = ing.strip().strip('"\'')
            if ing and ing.lower() not in ['ingredients not specified', 'nan', 'none', '']:
                cleaned_ingredients.append(ing)
        
        return cleaned_ingredients
    
    def _parse_instructions(self, instructions_text: str) -> List[str]:
        """Parse instructions from text into list of steps."""
        if not instructions_text or instructions_text.strip() == '':
            return []
        
        instructions_text = instructions_text.strip()
        
        # Split by common delimiters for steps
        separators = ['\n', '.', '|']
        
        # Try splitting by periods first (most common)
        steps = []
        if '.' in instructions_text:
            potential_steps = instructions_text.split('.')
            for step in potential_steps:
                step = step.strip()
                if len(step) > 10:  # Filter out very short fragments
                    steps.append(step)
        
        # If no good split by periods, try newlines
        if not steps and '\n' in instructions_text:
            steps = [step.strip() for step in instructions_text.split('\n') if step.strip()]
        
        # If still no good split, keep as single instruction
        if not steps:
            steps = [instructions_text]
        
        return steps
    
    def _save_training_data(self, recipes_data: List[Dict], output_path: str):
        """Save parsed recipes to training JSON format."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recipes_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(recipes_data)} recipes to {output_path}")
    
    def list_supported_formats(self) -> List[str]:
        """List all supported CSV formats."""
        return list(self.supported_formats.keys())
    
    def batch_process_directory(self, data_dir: str, output_dir: str = None):
        """Process all CSV files in a directory."""
        data_path = Path(data_dir)
        if output_dir is None:
            output_dir = data_dir
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        csv_files = list(data_path.glob("**/*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files in {data_dir}")
        
        for csv_file in csv_files:
            try:
                # Skip processed files
                if 'processed' in csv_file.name:
                    continue
                
                logger.info(f"Processing {csv_file}")
                
                # Generate output filename
                output_file = output_path / csv_file.parent.name / "training.json"
                output_file.parent.mkdir(exist_ok=True)
                
                # Parse the file
                recipes_data = self.parse_csv_to_training_format(str(csv_file), output_path=str(output_file))
                
                logger.info(f"Successfully processed {csv_file} -> {output_file}")
                
            except Exception as e:
                logger.error(f"Error processing {csv_file}: {e}")
                continue

def main():
    parser = argparse.ArgumentParser(description="Parse CSV recipe files to training format")
    parser.add_argument("csv_path", help="Path to CSV file to parse")
    parser.add_argument("--format", choices=['recipe_dataset_simple', 'recipe_box', 'indian_recipe_api', 
                                           'recipe_nlg', 'indian_food_analysis', 'allrecipes_ingredients'],
                       help="CSV format type (auto-detected if not specified)")
    parser.add_argument("--output", help="Output JSON file path")
    parser.add_argument("--batch-dir", help="Process all CSV files in directory")
    parser.add_argument("--list-formats", action="store_true", help="List supported formats")
    
    args = parser.parse_args()
    
    parser_obj = CSVRecipeParser()
    
    if args.list_formats:
        print("Supported CSV formats:")
        for fmt in parser_obj.list_supported_formats():
            print(f"  - {fmt}")
        return
    
    if args.batch_dir:
        parser_obj.batch_process_directory(args.batch_dir)
        return
    
    # Single file processing
    if not os.path.exists(args.csv_path):
        print(f"Error: CSV file not found: {args.csv_path}")
        return
    
    try:
        recipes_data = parser_obj.parse_csv_to_training_format(
            args.csv_path, 
            format_type=args.format,
            output_path=args.output
        )
        
        print(f"Successfully parsed {len(recipes_data)} recipes")
        if args.output:
            print(f"Saved to: {args.output}")
        else:
            print("Preview of first recipe:")
            if recipes_data:
                print(json.dumps(recipes_data[0], indent=2))
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()