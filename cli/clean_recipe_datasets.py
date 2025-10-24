#!/usr/bin/env python3
"""
Recipe Dataset Cleaner - Remove garbage, fix common issues, validate quality
Handles both JSON and CSV formats
"""

import json
import csv
import re
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from collections import Counter


class RecipeDataCleaner:
    """Clean and validate recipe datasets"""

    def __init__(self, verbose: bool = False, dry_run: bool = False):
        self.verbose = verbose
        self.dry_run = dry_run

        # Statistics
        self.stats = {
            'total_processed': 0,
            'removed_garbage': 0,
            'removed_too_short': 0,
            'removed_missing_fields': 0,
            'removed_duplicates': 0,
            'fixed_formatting': 0,
            'kept_clean': 0
        }

        # Garbage patterns
        self.garbage_patterns = [
            r':\)+',  # :))) :))
            r':\(+',  # :((( :((
            r'\blol\b',
            r'\bhaha+\b',
            r'\bwtf\b',
            r'\bomg\b',
            r'\btest\b.*recipe',
            r'\bgarbage\b',
            r'\bxxx+\b',
            r'\basdf+\b',
            r'\b(fuck|shit|damn)\b',
            r'http[s]?://(?!.*recipe)',  # URLs that aren't recipe-related
            r'\b\d{10,}\b',  # Long number sequences (phone numbers, etc)
            r'[!?]{3,}',  # Multiple exclamation/question marks
            r'\.{4,}',  # Excessive periods
            r'(\w)\1{5,}',  # Repeated characters (aaaaaaa)
        ]

        # Excel date conversion issues (common in recipe datasets)
        self.excel_date_fixes = {
            r'\b(\d+)-Jan\b': lambda m: f'{int(m.group(1))}/2',  # 2-Jan -> 1/2
            r'\b(\d+)-Feb\b': lambda m: f'{int(m.group(1))}/3',  # 2-Feb -> 2/3
            r'\b(\d+)-Mar\b': lambda m: f'{int(m.group(1))}/4',
            r'\b(\d+)-Apr\b': lambda m: f'{int(m.group(1))}/5',
            r'\b(\d+)-May\b': lambda m: f'{int(m.group(1))}/6',
            r'\b(\d+)-Jun\b': lambda m: f'{int(m.group(1))}/7',
            r'\b(\d+)-Jul\b': lambda m: f'{int(m.group(1))}/8',
        }

        # Duplicate tracking
        self.seen_recipes = set()

    def is_garbage(self, text: str) -> Tuple[bool, str]:
        """Check if text contains garbage patterns"""
        if not text or not isinstance(text, str):
            return True, "empty or non-string"

        text_lower = text.lower()

        # Check garbage patterns
        for pattern in self.garbage_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True, f"matches garbage pattern: {pattern}"

        # Check for excessive special characters
        special_char_ratio = len(re.findall(r'[^a-zA-Z0-9\s\.,!\?;:\-\'\"()\[\]]', text)) / max(len(text), 1)
        if special_char_ratio > 0.3:
            return True, f"too many special characters ({special_char_ratio:.1%})"

        # Check for minimum meaningful content
        words = re.findall(r'\b[a-zA-Z]{2,}\b', text)
        if len(words) < 3:
            return True, "too few words"

        # Check for excessive capitalization
        if len(text) > 10:
            caps_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if caps_ratio > 0.8:
                return True, f"excessive caps ({caps_ratio:.1%})"

        return False, ""

    def fix_excel_dates(self, text: str) -> str:
        """Fix Excel date conversion issues in ingredient quantities"""
        if not isinstance(text, str):
            return text

        fixed = text
        for pattern, replacement in self.excel_date_fixes.items():
            fixed = re.sub(pattern, replacement, fixed, flags=re.IGNORECASE)

        return fixed

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not isinstance(text, str):
            return str(text) if text else ""

        # Fix Excel dates
        text = self.fix_excel_dates(text)

        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Fix common encoding issues
        text = text.replace('\u00b0', '°')  # Degree symbol
        text = text.replace('\u00bd', '1/2')
        text = text.replace('\u00bc', '1/4')
        text = text.replace('\u00be', '3/4')

        # Remove zero-width characters
        text = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', text)

        return text.strip()

    def normalize_list(self, items: Any) -> List[str]:
        """Normalize list items (could be string, list, or JSON string)"""
        if isinstance(items, str):
            # Try parsing as JSON first
            try:
                items = json.loads(items)
            except:
                # Split by common delimiters
                items = re.split(r'[,;\n]', items)

        if not isinstance(items, list):
            items = [str(items)]

        # Clean each item
        cleaned = []
        for item in items:
            cleaned_item = self.clean_text(str(item))
            if cleaned_item and not self.is_garbage(cleaned_item)[0]:
                cleaned.append(cleaned_item)

        return cleaned

    def get_recipe_fingerprint(self, recipe: Dict[str, Any]) -> str:
        """Create fingerprint for duplicate detection"""
        # Use title + first 3 ingredients as fingerprint
        title = str(recipe.get('title', recipe.get('name', ''))).lower().strip()

        ingredients = recipe.get('ingredients', recipe.get('Ingredients', []))
        if isinstance(ingredients, str):
            ing_str = ingredients[:100].lower()
        elif isinstance(ingredients, list):
            ing_str = ' '.join(str(i).lower() for i in ingredients[:3])
        else:
            ing_str = ''

        return f"{title}::{ing_str}"

    def validate_recipe(self, recipe: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate a recipe meets minimum quality standards"""

        # Check for required fields (flexible field names)
        title_fields = ['title', 'name', 'recipe_name', 'Title', 'RecipeName']
        ingredients_fields = ['ingredients', 'Ingredients', 'ingredient_tokens']
        instructions_fields = ['instructions', 'directions', 'steps', 'Instructions']

        has_title = any(recipe.get(f) for f in title_fields)
        has_ingredients = any(recipe.get(f) for f in ingredients_fields)
        has_instructions = any(recipe.get(f) for f in instructions_fields)

        if not has_title:
            return False, "missing title"

        if not has_ingredients and not has_instructions:
            return False, "missing both ingredients and instructions"

        # Get actual values
        title = next((recipe.get(f) for f in title_fields if recipe.get(f)), "")
        ingredients = next((recipe.get(f) for f in ingredients_fields if recipe.get(f)), [])
        instructions = next((recipe.get(f) for f in instructions_fields if recipe.get(f)), [])

        # Validate title
        is_garbage, reason = self.is_garbage(str(title))
        if is_garbage:
            return False, f"garbage title: {reason}"

        # Validate ingredients
        if ingredients:
            ing_list = self.normalize_list(ingredients)
            if len(ing_list) < 2:
                return False, "too few ingredients"

            # Check each ingredient
            for ing in ing_list[:5]:  # Check first 5
                is_garbage, reason = self.is_garbage(ing)
                if is_garbage:
                    return False, f"garbage ingredient: {reason}"

        # Validate instructions
        if instructions:
            inst_list = self.normalize_list(instructions)
            inst_text = ' '.join(inst_list)

            if len(inst_text) < 20:
                return False, "instructions too short"

            is_garbage, reason = self.is_garbage(inst_text)
            if is_garbage:
                return False, f"garbage instructions: {reason}"

        # Check for duplicates
        fingerprint = self.get_recipe_fingerprint(recipe)
        if fingerprint in self.seen_recipes:
            return False, "duplicate recipe"

        self.seen_recipes.add(fingerprint)

        return True, "valid"

    def clean_recipe(self, recipe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Clean a single recipe"""
        self.stats['total_processed'] += 1

        # Validate first
        is_valid, reason = self.validate_recipe(recipe)

        if not is_valid:
            if 'garbage' in reason:
                self.stats['removed_garbage'] += 1
            elif 'too short' in reason or 'too few' in reason:
                self.stats['removed_too_short'] += 1
            elif 'missing' in reason:
                self.stats['removed_missing_fields'] += 1
            elif 'duplicate' in reason:
                self.stats['removed_duplicates'] += 1

            if self.verbose:
                title = recipe.get('title', recipe.get('name', 'Unknown'))
                print(f"  ❌ Rejected: {title[:50]} - {reason}")

            return None

        # Clean all text fields
        cleaned = {}
        needs_fixing = False

        for key, value in recipe.items():
            if isinstance(value, str):
                cleaned_value = self.clean_text(value)
                if cleaned_value != value:
                    needs_fixing = True
                cleaned[key] = cleaned_value
            elif isinstance(value, list):
                cleaned_list = self.normalize_list(value)
                if cleaned_list != value:
                    needs_fixing = True
                cleaned[key] = cleaned_list
            else:
                cleaned[key] = value

        if needs_fixing:
            self.stats['fixed_formatting'] += 1

        self.stats['kept_clean'] += 1
        return cleaned

    def clean_json_file(self, input_path: Path, output_path: Optional[Path] = None) -> int:
        """Clean a JSON/JSONL file"""
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"

        print(f"\n🧹 Cleaning: {input_path}")

        cleaned_recipes = []

        # Try as JSONL first
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        recipe = json.loads(line)
                        cleaned = self.clean_recipe(recipe)
                        if cleaned:
                            cleaned_recipes.append(cleaned)
                    except json.JSONDecodeError:
                        if self.verbose:
                            print(f"  ⚠️  Line {line_num}: Invalid JSON")
                        continue
        except Exception as e:
            # Try as regular JSON array
            try:
                with open(input_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                    if isinstance(data, list):
                        for recipe in data:
                            cleaned = self.clean_recipe(recipe)
                            if cleaned:
                                cleaned_recipes.append(cleaned)
                    elif isinstance(data, dict):
                        cleaned = self.clean_recipe(data)
                        if cleaned:
                            cleaned_recipes.append(cleaned)
            except Exception as e:
                print(f"  ❌ Error reading file: {e}")
                return 0

        # Write cleaned data
        if not self.dry_run and cleaned_recipes:
            with open(output_path, 'w', encoding='utf-8') as f:
                # Write as JSONL for efficiency
                for recipe in cleaned_recipes:
                    f.write(json.dumps(recipe, ensure_ascii=False) + '\n')

            print(f"  ✅ Saved {len(cleaned_recipes)} clean recipes to: {output_path}")

        return len(cleaned_recipes)

    def clean_csv_file(self, input_path: Path, output_path: Optional[Path] = None) -> int:
        """Clean a CSV file"""
        if output_path is None:
            output_path = input_path.parent / f"{input_path.stem}_cleaned{input_path.suffix}"

        print(f"\n🧹 Cleaning: {input_path}")

        cleaned_recipes = []
        headers = None

        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']

        for encoding in encodings:
            try:
                with open(input_path, 'r', encoding=encoding, newline='') as f:
                    reader = csv.DictReader(f)
                    headers = reader.fieldnames

                    for row in reader:
                        cleaned = self.clean_recipe(row)
                        if cleaned:
                            cleaned_recipes.append(cleaned)

                break  # Success
            except (UnicodeDecodeError, UnicodeError):
                continue
            except Exception as e:
                print(f"  ❌ Error reading CSV: {e}")
                return 0

        # Write cleaned data
        if not self.dry_run and cleaned_recipes and headers:
            with open(output_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=headers)
                writer.writeheader()
                writer.writerows(cleaned_recipes)

            print(f"  ✅ Saved {len(cleaned_recipes)} clean recipes to: {output_path}")

        return len(cleaned_recipes)

    def clean_dataset(self, input_path: Path, output_path: Optional[Path] = None) -> int:
        """Clean a dataset file (auto-detect format)"""
        if input_path.suffix.lower() in ['.json', '.jsonl']:
            return self.clean_json_file(input_path, output_path)
        elif input_path.suffix.lower() == '.csv':
            return self.clean_csv_file(input_path, output_path)
        else:
            print(f"  ⚠️  Unsupported format: {input_path.suffix}")
            return 0

    def print_stats(self):
        """Print cleaning statistics"""
        print("\n" + "=" * 60)
        print("📊 CLEANING STATISTICS")
        print("=" * 60)
        print(f"Total recipes processed:     {self.stats['total_processed']:,}")
        print(f"Kept (clean):               {self.stats['kept_clean']:,} ✅")
        print(f"Fixed formatting issues:    {self.stats['fixed_formatting']:,} 🔧")
        print(f"Removed (garbage):          {self.stats['removed_garbage']:,} 🗑️")
        print(f"Removed (too short):        {self.stats['removed_too_short']:,} 📏")
        print(f"Removed (missing fields):   {self.stats['removed_missing_fields']:,} ❓")
        print(f"Removed (duplicates):       {self.stats['removed_duplicates']:,} 🔁")
        print()

        total_removed = (self.stats['removed_garbage'] +
                        self.stats['removed_too_short'] +
                        self.stats['removed_missing_fields'] +
                        self.stats['removed_duplicates'])

        if self.stats['total_processed'] > 0:
            kept_pct = (self.stats['kept_clean'] / self.stats['total_processed']) * 100
            print(f"Quality rate: {kept_pct:.1f}% kept, {100-kept_pct:.1f}% removed")


def main():
    parser = argparse.ArgumentParser(
        description='Clean recipe datasets by removing garbage data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Clean a single file
  python clean_recipe_datasets.py data/training.json

  # Clean all datasets in a directory
  python clean_recipe_datasets.py data/datasets/ --recursive

  # Dry run (preview what would be cleaned)
  python clean_recipe_datasets.py data/datasets/ -r --dry-run -v

  # Clean and specify output directory
  python clean_recipe_datasets.py data/datasets/ -r -o data/cleaned_datasets/
        """
    )

    parser.add_argument('input', type=str, help='Input file or directory')
    parser.add_argument('-o', '--output', type=str, help='Output file or directory')
    parser.add_argument('-r', '--recursive', action='store_true',
                       help='Process directory recursively')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run - show what would be done without writing files')

    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    cleaner = RecipeDataCleaner(verbose=args.verbose, dry_run=args.dry_run)

    if args.dry_run:
        print("🔍 DRY RUN MODE - No files will be modified\n")

    # Process files
    if input_path.is_file():
        cleaner.clean_dataset(input_path, output_path)
    elif input_path.is_dir():
        # Find all JSON and CSV files
        pattern = '**/*' if args.recursive else '*'

        json_files = list(input_path.glob(f'{pattern}.json'))
        jsonl_files = list(input_path.glob(f'{pattern}.jsonl'))
        csv_files = list(input_path.glob(f'{pattern}.csv'))

        all_files = json_files + jsonl_files + csv_files

        print(f"📁 Found {len(all_files)} dataset files to clean")

        for file_path in all_files:
            # Skip tiny files
            if file_path.stat().st_size < 1024:  # Skip files < 1KB
                continue

            # Determine output path
            if output_path and output_path.is_dir():
                relative = file_path.relative_to(input_path)
                file_output = output_path / relative.parent / f"{relative.stem}_cleaned{relative.suffix}"
                file_output.parent.mkdir(parents=True, exist_ok=True)
            else:
                file_output = None

            cleaner.clean_dataset(file_path, file_output)
    else:
        print(f"❌ Invalid input path: {input_path}")
        return 1

    # Print final statistics
    cleaner.print_stats()

    return 0


if __name__ == "__main__":
    sys.exit(main())
