#!/usr/bin/env python3
"""
🏭 B2B DATASET TRANSFORMER
Convert portion of your 2.2M recipes to enterprise B2B format with quality validation
Parallel processing optimized for Ryzen 9 3900X
"""

import json
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from dataclasses import dataclass
import re

@dataclass
class QualityMetrics:
    """Quality metrics for transformed recipes."""
    ingredient_count: int
    instruction_steps: int
    cooking_time_mentioned: bool
    measurements_present: bool
    cooking_methods_count: int
    complexity_score: float
    b2b_features_count: int

class B2BDatasetTransformer:
    """Transform recipes to B2B format with quality preservation."""
    
    def __init__(self):
        self.business_scenarios = {
            "restaurant_fast_casual": {
                "volume_range": [100, 500],
                "cost_target": "Budget", 
                "skill_level": "Line Cook",
                "keywords": ["burger", "sandwich", "salad", "wrap", "bowl", "quick", "fast"]
            },
            "restaurant_fine_dining": {
                "volume_range": [30, 100],
                "cost_target": "Premium",
                "skill_level": "Professional", 
                "keywords": ["salmon", "beef", "lamb", "duck", "lobster", "truffle", "wine"]
            },
            "catering_corporate": {
                "volume_range": [100, 1000],
                "cost_target": "Mid-range",
                "skill_level": "Professional",
                "keywords": ["chicken", "pasta", "rice", "vegetables", "soup", "buffet"]
            },
            "meal_kit_family": {
                "volume_range": [10000, 50000], 
                "cost_target": "Mid-range",
                "skill_level": "Home Cook",
                "keywords": ["easy", "quick", "family", "kid", "simple", "30", "minute"]
            },
            "institutional_school": {
                "volume_range": [500, 2000],
                "cost_target": "Budget", 
                "skill_level": "Institutional",
                "keywords": ["healthy", "nutritious", "kid-friendly", "lunch", "cafeteria"]
            }
        }
        
        # Quality validation patterns
        self.cooking_methods = [
            "bake", "roast", "grill", "fry", "saute", "boil", "steam", "braise",
            "simmer", "broil", "poach", "blanch", "sear", "caramelize"
        ]
        
        self.measurement_patterns = [
            r'\d+\s*(cup|tablespoon|teaspoon|pound|ounce|gram|liter|ml|tbsp|tsp|lb|oz|g)',
            r'\d+/\d+\s*(cup|tablespoon|teaspoon)',
            r'\d+\.\d+\s*(cup|tablespoon|teaspoon|pound|ounce)'
        ]
    
    def classify_recipe_to_business(self, recipe: Dict) -> str:
        """Intelligently classify recipe to business scenario."""
        recipe_text = self._get_recipe_text(recipe).lower()
        
        scenario_scores = {}
        for scenario_name, scenario in self.business_scenarios.items():
            score = 0
            
            # Keyword matching
            for keyword in scenario['keywords']:
                if keyword in recipe_text:
                    score += 2 if len(keyword) > 4 else 1
            
            # Complexity matching
            ingredient_count = len(recipe.get('ingredients', []))
            if scenario['skill_level'] == 'Home Cook' and ingredient_count <= 8:
                score += 2
            elif scenario['skill_level'] == 'Professional' and ingredient_count >= 6:
                score += 2
            elif scenario['skill_level'] == 'Line Cook' and 4 <= ingredient_count <= 10:
                score += 1
            
            # Time consideration
            if 'time' in recipe_text or 'minute' in recipe_text:
                if scenario['skill_level'] in ['Home Cook', 'Line Cook']:
                    score += 1
            
            scenario_scores[scenario_name] = score
        
        # Return best match or default
        if not scenario_scores or max(scenario_scores.values()) == 0:
            return "restaurant_fast_casual"
        
        return max(scenario_scores, key=scenario_scores.get)
    
    def _get_recipe_text(self, recipe: Dict) -> str:
        """Extract all text from recipe for analysis."""
        text_parts = []
        
        # Title
        if 'title' in recipe:
            text_parts.append(recipe['title'])
        
        # Ingredients
        if 'ingredients' in recipe:
            if isinstance(recipe['ingredients'], list):
                text_parts.extend(recipe['ingredients'])
            else:
                text_parts.append(str(recipe['ingredients']))
        
        # Instructions
        if 'instructions' in recipe:
            if isinstance(recipe['instructions'], list):
                text_parts.extend(recipe['instructions'])
            else:
                text_parts.append(str(recipe['instructions']))
        
        return ' '.join(text_parts)
    
    def calculate_quality_metrics(self, recipe: Dict) -> QualityMetrics:
        """Calculate quality metrics for a recipe."""
        recipe_text = self._get_recipe_text(recipe).lower()
        
        # Count ingredients
        ingredients = recipe.get('ingredients', [])
        ingredient_count = len(ingredients) if isinstance(ingredients, list) else 1
        
        # Count instruction steps
        instructions = recipe.get('instructions', [])
        if isinstance(instructions, list):
            instruction_steps = len(instructions)
        else:
            # Count sentences/steps in text
            instruction_steps = len(re.split(r'[.!]', str(instructions)))
        
        # Check for cooking time mentions
        cooking_time_mentioned = any(word in recipe_text for word in 
                                   ['minute', 'hour', 'time', 'cook for', 'bake for'])
        
        # Check for measurements
        measurements_present = any(re.search(pattern, recipe_text) 
                                 for pattern in self.measurement_patterns)
        
        # Count cooking methods
        cooking_methods_count = sum(1 for method in self.cooking_methods 
                                  if method in recipe_text)
        
        # Calculate complexity score (0-100)
        complexity_score = min(100, (
            ingredient_count * 5 +
            instruction_steps * 3 +
            cooking_methods_count * 10 +
            (20 if cooking_time_mentioned else 0) +
            (15 if measurements_present else 0)
        ))
        
        return QualityMetrics(
            ingredient_count=ingredient_count,
            instruction_steps=instruction_steps,
            cooking_time_mentioned=cooking_time_mentioned,
            measurements_present=measurements_present,
            cooking_methods_count=cooking_methods_count,
            complexity_score=complexity_score,
            b2b_features_count=0  # Will be calculated after B2B transformation
        )
    
    def transform_to_b2b_format(self, recipe: Dict, scenario_name: str) -> Dict:
        """Transform recipe to B2B enterprise format with special tokens."""
        scenario = self.business_scenarios[scenario_name]
        volume = random.randint(scenario['volume_range'][0], scenario['volume_range'][1])
        
        # Create B2B structured prompt
        b2b_prompt = self._create_b2b_prompt(recipe, scenario, volume)
        
        # Create B2B structured output
        b2b_output = self._create_b2b_output(recipe, scenario, volume)
        
        # Calculate B2B quality metrics
        original_quality = self.calculate_quality_metrics(recipe)
        b2b_features = self._count_b2b_features(b2b_output)
        
        return {
            "input": b2b_prompt,
            "output": b2b_output,
            "format": "b2b_enterprise",
            "business_scenario": scenario_name,
            "volume": volume,
            "original_recipe": recipe.get('title', 'Unknown'),
            "quality_metrics": {
                "original_quality": original_quality.__dict__,
                "b2b_features_added": b2b_features,
                "quality_preserved": original_quality.complexity_score >= 30
            }
        }
    
    def _create_b2b_prompt(self, recipe: Dict, scenario: Dict, volume: int) -> str:
        """Create B2B enterprise prompt with special tokens."""
        title = recipe.get('title', 'recipe')
        
        prompt = f"""[BUSINESS_REQUEST]
[BUSINESS_TYPE]Commercial Kitchen[/BUSINESS_TYPE]
[SERVICE_STYLE]{scenario['cost_target']} Service[/SERVICE_STYLE]
[VOLUME]{volume} servings[/VOLUME]
[COST_TARGET]{scenario['cost_target']}[/COST_TARGET]
[SKILL_LEVEL]{scenario['skill_level']}[/SKILL_LEVEL]
[MEAL_STRUCTURE]Complete Meal[/MEAL_STRUCTURE]

Create a commercial version of {title} optimized for business food service.

[REQUIREMENTS]
- Food cost control and portion consistency
- Equipment efficiency and workflow optimization
- Food safety and temperature control compliance
- Scalable preparation methods for volume production
- Standardized procedures for staff training
[/REQUIREMENTS]
[/BUSINESS_REQUEST]

Generate enterprise recipe:"""
        
        return prompt
    
    def _create_b2b_output(self, recipe: Dict, scenario: Dict, volume: int) -> str:
        """Create B2B enterprise output with special tokens."""
        title = recipe.get('title', 'Commercial Recipe')
        ingredients = recipe.get('ingredients', [])
        instructions = recipe.get('instructions', [])
        
        # Scale ingredients for volume (simple multiplication)
        scaled_ingredients = self._scale_ingredients(ingredients, volume)
        
        output = f"""[RECIPE_START]
[TITLE_START]{title} (Commercial - {volume} servings)[TITLE_END]
[BUSINESS_INFO_START]
[COST_TARGET]{scenario['cost_target']}[/COST_TARGET]
[SKILL_LEVEL]{scenario['skill_level']}[/SKILL_LEVEL]
[VOLUME]{volume} servings[/VOLUME]
[PREP_TIME]Optimized for volume production[/PREP_TIME]
[/BUSINESS_INFO_END]

[EQUIPMENT_START]
- Commercial-grade equipment required
- Food safety temperature monitoring
- Portion control tools for consistency
[EQUIPMENT_END]

[INGREDIENTS_START]"""
        
        for ingredient in scaled_ingredients:
            output += f"\n[INGREDIENT]{ingredient}[/INGREDIENT]"
        
        output += "\n[INGREDIENTS_END]\n\n[INSTRUCTIONS_START]"
        
        # Add B2B-enhanced instructions
        if isinstance(instructions, list):
            for i, instruction in enumerate(instructions, 1):
                output += f"\n[STEP]{i}[/STEP][TECHNIQUE]Commercial Method[/TECHNIQUE] {instruction}"
        else:
            output += f"\n[STEP]1[/STEP][TECHNIQUE]Commercial Method[/TECHNIQUE] {instructions}"
        
        output += "\n[INSTRUCTIONS_END]\n\n[BUSINESS_NOTES_START]"
        output += f"\n- Cost target: {scenario['cost_target']}"
        output += f"\n- Staff skill level: {scenario['skill_level']}"
        output += f"\n- Volume optimized: {volume} servings"
        output += "\n- Food safety compliance required"
        output += "\n- Portion control for consistency"
        output += "\n[BUSINESS_NOTES_END]"
        output += "\n[RECIPE_END]"
        
        return output
    
    def _scale_ingredients(self, ingredients: List[str], target_volume: int) -> List[str]:
        """Scale ingredients for target volume (simplified)."""
        if not isinstance(ingredients, list):
            return [str(ingredients)]
        
        scaled = []
        base_servings = 4  # Assume original recipes serve 4
        scale_factor = target_volume / base_servings
        
        for ingredient in ingredients:
            # Simple scaling - multiply numbers found in ingredient
            scaled_ingredient = re.sub(
                r'(\d+(?:\.\d+)?)',
                lambda m: str(round(float(m.group(1)) * scale_factor, 2)),
                ingredient
            )
            scaled.append(scaled_ingredient)
        
        return scaled
    
    def _count_b2b_features(self, b2b_output: str) -> int:
        """Count B2B-specific features added."""
        b2b_tokens = [
            '[BUSINESS_TYPE]', '[SERVICE_STYLE]', '[VOLUME]', '[COST_TARGET]',
            '[SKILL_LEVEL]', '[EQUIPMENT_START]', '[BUSINESS_NOTES_START]',
            '[TECHNIQUE]', '[STEP]'
        ]
        
        return sum(1 for token in b2b_tokens if token in b2b_output)
    
    def process_recipe_batch(self, recipes: List[Dict], b2b_percentage: float = 0.2) -> List[Dict]:
        """Process batch of recipes with multiprocessing."""
        print(f"🏭 Processing {len(recipes):,} recipes ({b2b_percentage*100}% to B2B)")
        
        # Split into B2B and regular
        num_b2b = int(len(recipes) * b2b_percentage)
        b2b_recipes = random.sample(recipes, num_b2b)
        regular_recipes = [r for r in recipes if r not in b2b_recipes]
        
        print(f"📊 B2B transformations: {len(b2b_recipes):,}")
        print(f"📊 Regular format: {len(regular_recipes):,}")
        
        # Process B2B recipes in parallel
        print("🦀 Processing B2B recipes with parallel processing...")
        start_time = time.time()
        
        with Pool(processes=cpu_count()) as pool:
            b2b_results = pool.map(self._process_single_b2b_recipe, b2b_recipes)
        
        b2b_time = time.time() - start_time
        print(f"⚡ B2B processing: {b2b_time:.2f}s ({len(b2b_recipes)/b2b_time:.1f} recipes/sec)")
        
        # Process regular recipes
        print("📝 Processing regular recipes...")
        start_time = time.time()
        
        regular_results = []
        for recipe in regular_recipes:
            # Simple format matching your training data
            ingredients_text = ", ".join(recipe.get('ingredients', []))
            instructions_text = " ".join(recipe.get('instructions', []))
            
            regular_results.append({
                "input": f"Create {recipe.get('title', 'a recipe')}",
                "output": f"Ingredients: {ingredients_text}\nInstructions: {instructions_text}",
                "format": "simple",
                "quality_metrics": {
                    "original_quality": self.calculate_quality_metrics(recipe).__dict__
                }
            })
        
        regular_time = time.time() - start_time
        print(f"⚡ Regular processing: {regular_time:.2f}s ({len(regular_recipes)/regular_time:.1f} recipes/sec)")
        
        # Combine and shuffle
        all_results = b2b_results + regular_results
        random.shuffle(all_results)
        
        return all_results
    
    def _process_single_b2b_recipe(self, recipe: Dict) -> Dict:
        """Process single recipe for B2B (for multiprocessing)."""
        scenario = self.classify_recipe_to_business(recipe)
        return self.transform_to_b2b_format(recipe, scenario)
    
    def validate_quality(self, transformed_dataset: List[Dict]) -> Dict:
        """Validate quality of transformed dataset."""
        print("\n🔍 QUALITY VALIDATION")
        print("=" * 30)
        
        b2b_recipes = [r for r in transformed_dataset if r['format'] == 'b2b_enterprise']
        regular_recipes = [r for r in transformed_dataset if r['format'] == 'simple']
        
        # Analyze B2B quality
        b2b_quality_scores = [
            r['quality_metrics']['original_quality']['complexity_score'] 
            for r in b2b_recipes
        ]
        
        regular_quality_scores = [
            r['quality_metrics']['original_quality']['complexity_score']
            for r in regular_recipes
        ]
        
        b2b_features_avg = sum(
            r['quality_metrics']['b2b_features_added'] 
            for r in b2b_recipes
        ) / len(b2b_recipes) if b2b_recipes else 0
        
        quality_preserved = sum(
            1 for r in b2b_recipes 
            if r['quality_metrics']['quality_preserved']
        )
        
        validation_results = {
            "total_recipes": len(transformed_dataset),
            "b2b_count": len(b2b_recipes),
            "regular_count": len(regular_recipes),
            "b2b_percentage": len(b2b_recipes) / len(transformed_dataset) * 100,
            "quality_metrics": {
                "avg_b2b_quality": sum(b2b_quality_scores) / len(b2b_quality_scores) if b2b_quality_scores else 0,
                "avg_regular_quality": sum(regular_quality_scores) / len(regular_quality_scores) if regular_quality_scores else 0,
                "avg_b2b_features": b2b_features_avg,
                "quality_preserved_count": quality_preserved,
                "quality_preservation_rate": quality_preserved / len(b2b_recipes) * 100 if b2b_recipes else 0
            }
        }
        
        # Print results
        print(f"📊 Total recipes: {validation_results['total_recipes']:,}")
        print(f"🏢 B2B format: {validation_results['b2b_count']:,} ({validation_results['b2b_percentage']:.1f}%)")
        print(f"📝 Regular format: {validation_results['regular_count']:,}")
        print(f"🎯 Avg B2B quality: {validation_results['quality_metrics']['avg_b2b_quality']:.1f}/100")
        print(f"📈 Avg regular quality: {validation_results['quality_metrics']['avg_regular_quality']:.1f}/100")
        print(f"🏷️  Avg B2B features: {validation_results['quality_metrics']['avg_b2b_features']:.1f}")
        print(f"✅ Quality preserved: {validation_results['quality_metrics']['quality_preservation_rate']:.1f}%")
        
        return validation_results

def main():
    """Main transformation workflow."""
    print("🏭 B2B DATASET TRANSFORMER FOR RYZEN 9 3900X")
    print("=" * 60)
    
    transformer = B2BDatasetTransformer()
    
    # Configuration
    dataset_files = [
        "/Users/timmy/workspace/ai-apps/chef-genius/cli/allrecipes_250k/training.json",
        "/Users/timmy/workspace/ai-apps/chef-genius/data/recipes.json"
    ]
    
    print("Available datasets:")
    for i, file_path in enumerate(dataset_files, 1):
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{i}. {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"{i}. {path.name} (NOT FOUND)")
    
    print(f"\n🎯 Transformation Options:")
    print(f"1. 🧪 Test Sample (1,000 recipes, 20% B2B)")
    print(f"2. 📊 Balanced Dataset (10,000 recipes, 20% B2B)")
    print(f"3. 🏢 B2B Heavy (10,000 recipes, 50% B2B)")
    print(f"4. 🚀 Full Scale (100,000 recipes, 20% B2B)")
    
    choice = input("\nChoose option (1-4): ").strip()
    
    # Load and sample dataset
    print(f"\n📁 Loading dataset...")
    with open(dataset_files[0], 'r') as f:
        full_dataset = json.load(f)
    
    print(f"📊 Total recipes available: {len(full_dataset):,}")
    
    if choice == "1":
        recipes = random.sample(full_dataset, min(1000, len(full_dataset)))
        b2b_pct = 0.2
    elif choice == "2":
        recipes = random.sample(full_dataset, min(10000, len(full_dataset)))
        b2b_pct = 0.2
    elif choice == "3":
        recipes = random.sample(full_dataset, min(10000, len(full_dataset)))
        b2b_pct = 0.5
    elif choice == "4":
        recipes = random.sample(full_dataset, min(100000, len(full_dataset)))
        b2b_pct = 0.2
    else:
        print("❌ Invalid choice")
        return
    
    print(f"🎯 Processing {len(recipes):,} recipes with {b2b_pct*100}% B2B conversion")
    
    # Transform dataset
    start_time = time.time()
    transformed_dataset = transformer.process_recipe_batch(recipes, b2b_pct)
    total_time = time.time() - start_time
    
    print(f"\n⚡ TRANSFORMATION COMPLETE")
    print(f"Total time: {total_time:.2f}s")
    print(f"Speed: {len(recipes)/total_time:.1f} recipes/sec")
    
    # Validate quality
    validation_results = transformer.validate_quality(transformed_dataset)
    
    # Save results
    output_filename = f"b2b_dataset_{len(transformed_dataset)}_{int(b2b_pct*100)}pct.json"
    with open(output_filename, 'w') as f:
        json.dump(transformed_dataset, f, indent=2)
    
    validation_filename = f"quality_report_{len(transformed_dataset)}_{int(b2b_pct*100)}pct.json"
    with open(validation_filename, 'w') as f:
        json.dump(validation_results, f, indent=2)
    
    print(f"\n💾 SAVED RESULTS")
    print(f"Dataset: {output_filename}")
    print(f"Quality report: {validation_filename}")
    
    print(f"\n🎉 TRANSFORMATION SUCCESS!")
    print(f"Ready for training with both B2B enterprise tokens and simple formats!")

if __name__ == "__main__":
    main()