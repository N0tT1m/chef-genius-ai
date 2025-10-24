#!/usr/bin/env python3
"""
Recipe Generation Testing Script
Generates 15-20 diverse recipes covering common to edge cases with detailed output and Discord notifications
"""

import os
import sys
import time
import torch
import requests
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class RecipePrompt:
    """Individual recipe generation prompt."""
    name: str
    prompt: str
    category: str  # "common", "fusion", "dietary", "technique", "edge_case"
    difficulty: str = "normal"  # normal, challenging, extreme
    expected_features: List[str] = None  # Features we hope to see

class RecipeGenerator:
    """Advanced recipe generation and testing system."""
    
    def __init__(self, discord_webhook: str = None):
        self.discord_webhook = discord_webhook
        self.results = []
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create diverse recipe prompts
        self.recipe_prompts = self._create_recipe_prompts()
        
    def _create_recipe_prompts(self) -> List[RecipePrompt]:
        """Create 20 diverse recipe generation prompts."""
        
        prompts = []
        
        # COMMON RECIPES (5 prompts)
        prompts.extend([
            RecipePrompt(
                name="Classic Chocolate Chip Cookies",
                prompt="Create a foolproof chocolate chip cookie recipe that produces perfectly chewy cookies with crispy edges. Include exact measurements and baking tips.",
                category="common",
                difficulty="normal",
                expected_features=["flour", "butter", "chocolate chips", "baking", "temperature", "minutes"]
            ),
            RecipePrompt(
                name="Perfect Grilled Chicken",
                prompt="Design a grilled chicken breast recipe with a flavorful marinade that ensures juicy, tender results every time. Include grilling techniques and internal temperature.",
                category="common",
                difficulty="normal",
                expected_features=["chicken", "marinade", "grill", "temperature", "juicy", "seasoning"]
            ),
            RecipePrompt(
                name="Homemade Pizza Dough",
                prompt="Develop a pizza dough recipe that can be made ahead and produces a crispy yet chewy crust. Include kneading techniques and rising times.",
                category="common",
                difficulty="normal",
                expected_features=["flour", "yeast", "knead", "rise", "crispy", "chewy"]
            ),
            RecipePrompt(
                name="Fresh Garden Salad",
                prompt="Create a vibrant garden salad recipe with homemade vinaigrette that showcases seasonal vegetables and balanced flavors.",
                category="common",
                difficulty="normal",
                expected_features=["vegetables", "vinaigrette", "seasonal", "fresh", "dressing", "balanced"]
            ),
            RecipePrompt(
                name="Beef Stir-Fry",
                prompt="Design a quick beef stir-fry recipe with vegetables that maintains the meat's tenderness and creates a glossy sauce coating everything perfectly.",
                category="common",
                difficulty="normal",
                expected_features=["beef", "vegetables", "sauce", "tender", "quick", "coating"]
            )
        ])
        
        # FUSION CUISINE (4 prompts)
        prompts.extend([
            RecipePrompt(
                name="Korean-Mexican Fusion Tacos",
                prompt="Create innovative fusion tacos combining Korean bulgogi flavors with traditional Mexican taco elements, including kimchi slaw and gochujang crema.",
                category="fusion",
                difficulty="challenging",
                expected_features=["bulgogi", "tacos", "kimchi", "gochujang", "fusion", "Korean", "Mexican"]
            ),
            RecipePrompt(
                name="Italian-Thai Curry Pasta",
                prompt="Develop a unique pasta dish that marries Italian pasta techniques with Thai curry flavors, creating a creamy coconut-based sauce with fresh herbs.",
                category="fusion",
                difficulty="challenging",
                expected_features=["pasta", "curry", "coconut", "Thai", "Italian", "herbs", "creamy"]
            ),
            RecipePrompt(
                name="Japanese-French Ramen Burger",
                prompt="Design a gourmet burger using ramen noodle 'buns' with French cooking techniques for the patty and sophisticated garnishes.",
                category="fusion",
                difficulty="challenging",
                expected_features=["ramen", "burger", "French", "Japanese", "gourmet", "techniques", "garnishes"]
            ),
            RecipePrompt(
                name="Indian-Mediterranean Flatbread",
                prompt="Create a flatbread recipe combining Indian spices and cooking methods with Mediterranean toppings and olive oil-based preparations.",
                category="fusion",
                difficulty="challenging",
                expected_features=["flatbread", "Indian", "Mediterranean", "spices", "olive oil", "toppings"]
            )
        ])
        
        # DIETARY RESTRICTIONS (4 prompts)
        prompts.extend([
            RecipePrompt(
                name="Vegan Chocolate Avocado Mousse",
                prompt="Develop a rich, creamy chocolate mousse using avocado as the base that's indistinguishable from traditional dairy-based versions.",
                category="dietary",
                difficulty="challenging",
                expected_features=["vegan", "chocolate", "avocado", "mousse", "creamy", "dairy-free"]
            ),
            RecipePrompt(
                name="Keto Cauliflower Mac and Cheese",
                prompt="Create a satisfying mac and cheese substitute using cauliflower that maintains the comfort food appeal while staying under 10g carbs per serving.",
                category="dietary",
                difficulty="challenging",
                expected_features=["keto", "cauliflower", "cheese", "low-carb", "comfort food", "substitute"]
            ),
            RecipePrompt(
                name="Gluten-Free Sourdough Bread",
                prompt="Design a gluten-free sourdough bread recipe that develops proper tang and texture using alternative flours and fermentation techniques.",
                category="dietary",
                difficulty="extreme",
                expected_features=["gluten-free", "sourdough", "fermentation", "alternative flours", "tang", "texture"]
            ),
            RecipePrompt(
                name="Paleo Pad Thai",
                prompt="Recreate the classic Pad Thai using only paleo-approved ingredients, including spiralized vegetables and compliant sauces that maintain authentic flavors.",
                category="dietary",
                difficulty="challenging",
                expected_features=["paleo", "pad thai", "spiralized", "compliant", "authentic", "vegetables"]
            )
        ])
        
        # ADVANCED TECHNIQUES (4 prompts)
        prompts.extend([
            RecipePrompt(
                name="Sous Vide Salmon with Precision",
                prompt="Create a sous vide salmon recipe with exact time and temperature specifications, including complementary sauces and side dish pairings.",
                category="technique",
                difficulty="challenging",
                expected_features=["sous vide", "salmon", "temperature", "precise", "timing", "sauce"]
            ),
            RecipePrompt(
                name="Hand-Pulled Noodles",
                prompt="Develop a traditional hand-pulled noodle recipe with detailed stretching techniques and dough consistency guidelines for perfect texture.",
                category="technique",
                difficulty="extreme",
                expected_features=["hand-pulled", "noodles", "stretching", "dough", "technique", "texture"]
            ),
            RecipePrompt(
                name="Perfect Croissants from Scratch",
                prompt="Create a comprehensive croissant recipe including lamination techniques, proofing schedules, and troubleshooting guide for flaky, buttery results.",
                category="technique",
                difficulty="extreme",
                expected_features=["croissants", "lamination", "proofing", "flaky", "buttery", "technique"]
            ),
            RecipePrompt(
                name="Smoking Brisket Competition Style",
                prompt="Design a competition-worthy brisket recipe with wood selection, temperature control, wrapping techniques, and judging criteria considerations.",
                category="technique",
                difficulty="extreme",
                expected_features=["brisket", "smoking", "competition", "wood", "temperature", "wrapping"]
            )
        ])
        
        # EDGE CASES (3 prompts)
        prompts.extend([
            RecipePrompt(
                name="Edible Flower Garden Salad",
                prompt="Create an artistic salad featuring 10 different edible flowers with guidance on sourcing, preparation, and flavor profiles of each bloom.",
                category="edge_case",
                difficulty="extreme",
                expected_features=["edible flowers", "artistic", "sourcing", "flavor profiles", "preparation"]
            ),
            RecipePrompt(
                name="Liquid Nitrogen Ice Cream",
                prompt="Develop a molecular gastronomy ice cream recipe using liquid nitrogen with safety protocols and unique flavor combinations.",
                category="edge_case",
                difficulty="extreme",
                expected_features=["liquid nitrogen", "molecular gastronomy", "safety", "ice cream", "protocols"]
            ),
            RecipePrompt(
                name="Foraging-Based Survival Meal",
                prompt="Design a complete meal using only foraged ingredients that could be found in North American forests, including identification tips and preparation methods.",
                category="edge_case",
                difficulty="extreme",
                expected_features=["foraging", "survival", "identification", "forest", "preparation", "wild"]
            )
        ])
        
        return prompts
    
    def _post_process_recipe(self, recipe: str, prompt: RecipePrompt) -> str:
        """Post-process recipe to ensure it has proper structure."""
        recipe_lower = recipe.lower()
        
        # Check if recipe has proper sections
        has_ingredients_section = any(word in recipe_lower for word in [
            "## ingredients:", "ingredients:", "ingredient list:"
        ])
        
        has_instructions_section = any(word in recipe_lower for word in [
            "## instructions:", "instructions:", "## steps:", "steps:", "## directions:", "directions:"
        ])
        
        # If missing critical sections, try to add them
        if not has_ingredients_section and not has_instructions_section:
            # Recipe is likely unstructured, try to add basic structure
            lines = recipe.split('\n')
            structured_recipe = f"## {prompt.name}\n\n"
            
            # Look for ingredient-like lines (contain measurements or common ingredients)
            ingredient_lines = []
            instruction_lines = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                line_lower = line.lower()
                # Check if line looks like an ingredient
                if any(word in line_lower for word in [
                    "cup", "tablespoon", "teaspoon", "tsp", "tbsp", "oz", "lb", "gram", "kg", "ml", "clove", "pinch"
                ]) or any(line_lower.startswith(word) for word in ["- ", "• ", "* "]):
                    ingredient_lines.append(line)
                # Check if line looks like an instruction
                elif any(word in line_lower for word in [
                    "heat", "cook", "bake", "stir", "mix", "whisk", "add", "pour", "serve", "prepare"
                ]) or any(pattern in line_lower for pattern in ["1.", "2.", "3.", "step", "first", "then", "next"]):
                    instruction_lines.append(line)
                else:
                    # Add to instructions by default
                    instruction_lines.append(line)
            
            if ingredient_lines:
                structured_recipe += "## Ingredients:\n"
                for ingredient in ingredient_lines:
                    if not ingredient.startswith(("- ", "• ", "* ")):
                        structured_recipe += f"- {ingredient}\n"
                    else:
                        structured_recipe += f"{ingredient}\n"
                structured_recipe += "\n"
            
            if instruction_lines:
                structured_recipe += "## Instructions:\n"
                for i, instruction in enumerate(instruction_lines, 1):
                    if not instruction.lower().startswith(("step", f"{i}.")):
                        structured_recipe += f"{i}. {instruction}\n"
                    else:
                        structured_recipe += f"{instruction}\n"
            
            # Add notes section if recipe seems incomplete
            if len(structured_recipe.split()) < 50:
                structured_recipe += f"\n## Notes:\n- Adjust seasoning to taste\n- Cooking time may vary based on equipment\n"
            
            return structured_recipe
        
        return recipe
    
    def _quick_quality_check(self, recipe: str) -> float:
        """Quick quality assessment for regeneration decisions."""
        recipe_lower = recipe.lower()
        word_count = len(recipe.split())
        
        # Basic structural checks
        has_ingredients = any(word in recipe_lower for word in ["ingredients", "cup", "tablespoon", "teaspoon"])
        has_instructions = any(word in recipe_lower for word in ["instructions", "cook", "bake", "heat", "mix", "stir"])
        has_measurements = any(char.isdigit() for char in recipe)
        is_long_enough = word_count >= 50
        
        score = sum([has_ingredients, has_instructions, has_measurements, is_long_enough]) / 4.0
        return score
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model and tokenizer from checkpoint with auto-detection."""
        try:
            print(f"🔄 Loading model from: {checkpoint_path}")
            
            # If checkpoint_path points to config.json, use the directory instead
            if checkpoint_path.endswith('config.json'):
                checkpoint_path = os.path.dirname(checkpoint_path)
            
            # Verify the checkpoint directory exists and has required files
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")
            
            required_files = ['config.json', 'pytorch_model.bin']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(checkpoint_path, f))]
            if missing_files:
                print(f"⚠️ Missing files in checkpoint: {missing_files}")
                # Try with model.safetensors instead
                if not os.path.exists(os.path.join(checkpoint_path, 'model.safetensors')):
                    raise FileNotFoundError(f"Required model files not found in: {checkpoint_path}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Auto-detect model type
            config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)
            print(f"📋 Detected model type: {config.model_type}")
            
            # Load model based on type
            if config.model_type in ['t5', 'mt5', 'bart', 'pegasus', 'mbart']:
                # Seq2Seq models
                print(f"🔄 Loading as Seq2Seq model...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=True
                )
                self.model_type = "seq2seq"
            else:
                # Causal LM models (GPT-2, GPT-Neo, etc.)
                print(f"🔄 Loading as Causal Language model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=True
                )
                self.model_type = "causal"
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded successfully")
            print(f"   Device: {self.device}")
            print(f"   Model type: {type(self.model).__name__} ({self.model_type})")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            return False
    
    def generate_recipe(self, prompt: RecipePrompt, max_length: int = 600, generation_mode: str = "normal") -> Dict[str, Any]:
        """Generate a recipe with detailed metrics."""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        
        # Configure generation parameters based on mode
        if generation_mode == "greedy":
            # Deterministic, conservative generation
            gen_params = {
                "do_sample": False,  # Greedy decoding
                "temperature": None,  # Not used in greedy
                "top_p": None,
                "top_k": None,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "length_penalty": 1.0
            }
        elif generation_mode == "normal":
            # Balanced generation (current settings)
            gen_params = {
                "do_sample": True,
                "temperature": 0.5,
                "top_p": 0.85,
                "top_k": 35,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 4,
                "early_stopping": True,
                "length_penalty": 1.1
            }
        elif generation_mode == "creative":
            # High creativity, more diverse generation
            gen_params = {
                "do_sample": True,
                "temperature": 0.9,
                "top_p": 0.95,
                "top_k": 60,
                "repetition_penalty": 1.15,
                "no_repeat_ngram_size": 2,
                "early_stopping": False,
                "length_penalty": 0.9
            }
        else:
            raise ValueError(f"Unknown generation mode: {generation_mode}")
        
        try:
            if self.model_type == "seq2seq":
                # FLAN-T5 optimized generation with structured prompting
                formatted_prompt = f"""Create a complete recipe for: {prompt.prompt}

Please format your response with these sections:
## Ingredients:
[List all ingredients with measurements]

## Instructions:
[Step-by-step cooking instructions]

## Notes:
[Any cooking tips, timing, or serving suggestions]"""
                
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,  # Increased for structured prompts
                    padding=True
                ).to(self.device)
                
                # Build generation kwargs, filtering out None values
                generation_kwargs = {
                    "max_new_tokens": max_length + 200,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                generation_kwargs.update({k: v for k, v in gen_params.items() if v is not None})
                
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # For FLAN-T5, the output is the complete response, not including the input
                recipe = generated_text.strip()
                
                # Post-process and validate recipe structure
                recipe = self._post_process_recipe(recipe, prompt)
                    
            else:
                # Causal LM generation (GPT-2, etc.) with structured prompting
                formatted_prompt = f"""Recipe: {prompt.prompt}

## Ingredients:
"""
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                # Build generation kwargs, filtering out None values
                generation_kwargs = {
                    "max_new_tokens": max_length + 200,
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                generation_kwargs.update({k: v for k, v in gen_params.items() if v is not None})
                
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                recipe = generated_text[len(formatted_prompt):].strip()
                if not recipe:
                    recipe = generated_text.strip()
                
                # Post-process and validate recipe structure
                recipe = self._post_process_recipe(recipe, prompt)
            
            generation_time = time.time() - start_time
            
            # Calculate detailed metrics
            word_count = len(recipe.split())
            char_count = len(recipe)
            sentence_count = len([s for s in recipe.split('.') if s.strip()])
            tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
            
            return {
                "recipe": recipe,
                "generation_time": generation_time,
                "word_count": word_count,
                "char_count": char_count,
                "sentence_count": sentence_count,
                "tokens_generated": max(0, tokens_generated),
                "tokens_per_second": max(0, tokens_generated) / generation_time if generation_time > 0 else 0
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "generation_time": time.time() - start_time
            }
    
    def analyze_recipe_quality(self, recipe: str, prompt: RecipePrompt) -> Dict[str, Any]:
        """Analyze the quality and completeness of generated recipe."""
        
        recipe_lower = recipe.lower()
        
        # Check for expected features
        features_found = []
        features_missing = []
        
        if prompt.expected_features:
            for feature in prompt.expected_features:
                if feature.lower() in recipe_lower:
                    features_found.append(feature)
                else:
                    features_missing.append(feature)
        
        feature_coverage = len(features_found) / len(prompt.expected_features) if prompt.expected_features else 1.0
        
        # Enhanced recipe structure analysis
        has_ingredients_section = any(word in recipe_lower for word in [
            "## ingredients:", "ingredients:", "ingredient list:", "what you need:", "you will need:"
        ])
        
        has_measurements = any(word in recipe_lower for word in [
            "cup", "tablespoon", "teaspoon", "tsp", "tbsp", "oz", "ounce", "lb", "pound", 
            "gram", "kg", "liter", "ml", "clove", "pinch", "dash"
        ])
        
        has_ingredients = has_ingredients_section or has_measurements or any(word in recipe_lower for word in [
            "salt", "pepper", "oil", "water", "flour", "butter", "sugar", "onion", "garlic"
        ])
        
        has_instructions_section = any(word in recipe_lower for word in [
            "## instructions:", "instructions:", "## steps:", "steps:", "## directions:", 
            "directions:", "## method:", "method:", "## preparation:", "preparation:"
        ])
        
        has_cooking_verbs = any(word in recipe_lower for word in [
            "heat", "cook", "bake", "stir", "mix", "whisk", "fold", "prepare", "serve", 
            "season", "place", "remove", "add", "pour", "slice", "chop", "sauté", "fry", "boil"
        ])
        
        has_sequential_steps = any(pattern in recipe_lower for pattern in [
            "1.", "2.", "3.", "step 1", "step 2", "first", "then", "next", "finally"
        ])
        
        has_instructions = has_instructions_section or (has_cooking_verbs and has_sequential_steps)
        
        has_timing = any(word in recipe_lower for word in [
            "minutes", "hours", "cook", "bake", "simmer", "boil", "time", "until", 
            "for", "about", "preheat", "°f", "°c", "degrees", "temperature"
        ])
        
        has_quantities = any(char.isdigit() for char in recipe)
        
        has_techniques = any(word in recipe_lower for word in [
            "whisk", "fold", "sauté", "grill", "roast", "braise", "steam", "fry",
            "blend", "puree", "marinate", "season", "garnish", "dice", "mince"
        ])
        
        structure_score = sum([has_ingredients, has_instructions, has_timing, has_quantities, has_techniques]) / 5.0
        
        # Content quality assessment
        word_count = len(recipe.split())
        is_appropriate_length = 100 <= word_count <= 1000  # Reasonable recipe length
        
        # Creativity and detail assessment
        creativity_indicators = [
            "innovative", "unique", "creative", "fusion", "twist", "variation",
            "gourmet", "artisanal", "signature", "special", "secret", "perfect"
        ]
        has_creativity = any(indicator in recipe_lower for indicator in creativity_indicators)
        
        # Technical detail assessment
        technical_terms = [
            "temperature", "internal", "consistency", "texture", "technique", 
            "method", "precise", "exact", "careful", "gentle", "vigorous"
        ]
        has_technical_detail = any(term in recipe_lower for term in technical_terms)
        
        # Calculate overall quality score
        quality_components = {
            "feature_coverage": feature_coverage * 0.25,
            "structure": structure_score * 0.30,
            "length": (1.0 if is_appropriate_length else 0.6) * 0.20,
            "creativity": (1.0 if has_creativity else 0.7) * 0.15,
            "technical_detail": (1.0 if has_technical_detail else 0.8) * 0.10
        }
        
        overall_quality = sum(quality_components.values())
        
        return {
            "overall_quality": overall_quality,
            "feature_coverage": feature_coverage,
            "structure_score": structure_score,
            "features_found": features_found,
            "features_missing": features_missing,
            "has_ingredients": has_ingredients,
            "has_instructions": has_instructions,
            "has_timing": has_timing,
            "has_quantities": has_quantities,
            "has_techniques": has_techniques,
            "has_creativity": has_creativity,
            "has_technical_detail": has_technical_detail,
            "is_appropriate_length": is_appropriate_length,
            "word_count": word_count,
            "quality_components": quality_components
        }
    
    def generate_single_recipe(self, prompt: RecipePrompt) -> Dict[str, Any]:
        """Generate and analyze a single recipe with 3 generation modes."""
        print(f"🍳 Generating: {prompt.name} ({prompt.category.upper()})")
        
        start_time = time.time()
        
        # Generate 3 versions: greedy, normal, creative
        generation_modes = ["greedy", "normal", "creative"]
        mode_results = {}
        
        for mode in generation_modes:
            print(f"  🎯 {mode.upper()} mode...")
            mode_result = self.generate_recipe(prompt, generation_mode=mode)
            
            if "error" in mode_result:
                mode_results[mode] = {
                    "status": "FAILED",
                    "error": mode_result["error"],
                    "generation_time": mode_result.get("generation_time", 0)
                }
                continue
            
            # Analyze quality for this mode
            quality_result = self.analyze_recipe_quality(mode_result["recipe"], prompt)
            
            mode_results[mode] = {
                "recipe": mode_result["recipe"],
                "status": "SUCCESS" if quality_result["overall_quality"] >= 0.4 else "NEEDS_IMPROVEMENT",
                "generation_time": mode_result["generation_time"],
                "overall_quality": quality_result["overall_quality"],
                "structure_score": quality_result["structure_score"],
                "feature_coverage": quality_result["feature_coverage"],
                "word_count": mode_result["word_count"],
                "tokens_per_second": mode_result["tokens_per_second"],
                "features_found": quality_result["features_found"],
                "features_missing": quality_result["features_missing"],
                "quality_components": quality_result["quality_components"]
            }
            
            status_emoji = "✅" if mode_results[mode]["status"] == "SUCCESS" else "⚠️"
            print(f"    {status_emoji} Quality: {quality_result['overall_quality']:.3f}")
        
        # Determine best overall result
        successful_modes = [mode for mode, result in mode_results.items() if result.get("status") == "SUCCESS"]
        if successful_modes:
            best_mode = max(successful_modes, key=lambda m: mode_results[m]["overall_quality"])
        else:
            best_mode = max(mode_results.keys(), key=lambda m: mode_results[m].get("overall_quality", 0))
        
        total_time = time.time() - start_time
        
        # Determine overall success based on best mode
        quality_thresholds = {
            "common": 0.6,
            "fusion": 0.5,
            "dietary": 0.5,
            "technique": 0.4,
            "edge_case": 0.3
        }
        
        threshold = quality_thresholds.get(prompt.category, 0.5)
        best_quality = mode_results[best_mode].get("overall_quality", 0)
        overall_status = "SUCCESS" if best_quality >= threshold else "NEEDS_IMPROVEMENT"
        
        result = {
            "name": prompt.name,
            "category": prompt.category,
            "difficulty": prompt.difficulty,
            "prompt": prompt.prompt,
            "status": overall_status,
            "total_time": total_time,
            "best_mode": best_mode,
            "mode_results": mode_results,
            # Best mode results for compatibility
            "recipe": mode_results[best_mode].get("recipe", ""),
            "generation_time": sum(r.get("generation_time", 0) for r in mode_results.values()),
            "overall_quality": best_quality,
            "structure_score": mode_results[best_mode].get("structure_score", 0),
            "feature_coverage": mode_results[best_mode].get("feature_coverage", 0),
            "word_count": mode_results[best_mode].get("word_count", 0),
            "tokens_per_second": sum(r.get("tokens_per_second", 0) for r in mode_results.values()) / len(mode_results),
            "features_found": mode_results[best_mode].get("features_found", []),
            "features_missing": mode_results[best_mode].get("features_missing", []),
            "quality_components": mode_results[best_mode].get("quality_components", {})
        }
        
        return result
    
    def generate_all_recipes(self, checkpoint_path: str, parallel_generations: int = 2) -> Dict[str, Any]:
        """Generate all recipes and compile comprehensive results."""
        
        print(f"\n🚀 RECIPE GENERATION SESSION")
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"🍳 Total recipes to generate: {len(self.recipe_prompts)}")
        print(f"⚡ Parallel generations: {parallel_generations}")
        print("=" * 80)
        
        # Load model
        if not self.load_model_from_checkpoint(checkpoint_path):
            return {"error": "Failed to load model"}
        
        # Generate recipes with some parallelization
        start_time = time.time()
        results = []
        
        # Group by category for organized output
        categories = {}
        for prompt in self.recipe_prompts:
            if prompt.category not in categories:
                categories[prompt.category] = []
            categories[prompt.category].append(prompt)
        
        # Generate recipes by category
        for category, prompts in categories.items():
            print(f"\n🎯 Generating {category.upper()} recipes ({len(prompts)} items)...")
            
            # Use limited parallelization to avoid memory issues
            with ThreadPoolExecutor(max_workers=min(parallel_generations, 2)) as executor:
                future_to_prompt = {executor.submit(self.generate_single_recipe, prompt): prompt for prompt in prompts}
                
                for future in as_completed(future_to_prompt):
                    result = future.result()
                    results.append(result)
                    
                    status_emoji = "✅" if result["status"] == "SUCCESS" else "⚠️"
                    quality_emoji = "🏆" if result["overall_quality"] > 0.8 else "🎯" if result["overall_quality"] > 0.6 else "📈"
                    
                    print(f"  {status_emoji} {result['name']}: {quality_emoji} {result['overall_quality']:.3f} quality")
        
        total_time = time.time() - start_time
        
        # Compile comprehensive statistics
        successful_results = [r for r in results if r["status"] == "SUCCESS"]
        
        # Category-wise statistics
        category_stats = {}
        for category in categories.keys():
            cat_results = [r for r in results if r["category"] == category]
            cat_successful = [r for r in cat_results if r["status"] == "SUCCESS"]
            
            category_stats[category] = {
                "total": len(cat_results),
                "successful": len(cat_successful),
                "success_rate": len(cat_successful) / len(cat_results) if cat_results else 0,
                "avg_quality": sum(r["overall_quality"] for r in cat_results) / len(cat_results) if cat_results else 0
            }
        
        comprehensive_results = {
            "checkpoint_path": checkpoint_path,
            "total_recipes": len(results),
            "successful_recipes": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "avg_quality_score": sum(r["overall_quality"] for r in results) / len(results) if results else 0,
            "avg_generation_time": sum(r["generation_time"] for r in results) / len(results) if results else 0,
            "avg_tokens_per_second": sum(r["tokens_per_second"] for r in results if r.get("tokens_per_second")) / len([r for r in results if r.get("tokens_per_second")]) if results else 0,
            "total_generation_time": total_time,
            "category_stats": category_stats,
            "individual_results": results
        }
        
        return comprehensive_results
    
    def send_discord_notification(self, results: Dict[str, Any]):
        """Send comprehensive recipe generation results to Discord."""
        
        if not self.discord_webhook:
            print("⚠️ No Discord webhook provided, skipping notification")
            return
        
        try:
            # Determine overall performance
            success_rate = results["success_rate"]
            avg_quality = results["avg_quality_score"]
            
            if success_rate >= 0.8 and avg_quality >= 0.7:
                color = 0x00ff00  # Green - excellent
                status_emoji = "🏆"
                status_text = "EXCELLENT PERFORMANCE"
            elif success_rate >= 0.6 and avg_quality >= 0.5:
                color = 0xffaa00  # Orange - good
                status_emoji = "✅"
                status_text = "GOOD PERFORMANCE"
            else:
                color = 0xff0000  # Red - needs work
                status_emoji = "⚠️"
                status_text = "NEEDS IMPROVEMENT"
            
            embed = {
                "title": f"{status_emoji} Recipe Generation Test Results",
                "description": f"**{status_text}** - Generated {results['total_recipes']} diverse recipes",
                "color": color,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "fields": [
                    {
                        "name": "📊 Overall Results",
                        "value": f"✅ Successful: {results['successful_recipes']}/{results['total_recipes']}\n📈 Success Rate: {success_rate:.1%}\n🎯 Avg Quality: {avg_quality:.3f}",
                        "inline": True
                    },
                    {
                        "name": "⚡ Performance",
                        "value": f"🏃 Avg Gen Time: {results['avg_generation_time']:.2f}s\n🚀 Tokens/sec: {results['avg_tokens_per_second']:.1f}\n⏱️ Total Time: {results['total_generation_time']:.1f}s",
                        "inline": True
                    },
                    {
                        "name": "📁 Model Tested",
                        "value": f"`{os.path.basename(results['checkpoint_path'])}`",
                        "inline": False
                    }
                ]
            }
            
            # Add category breakdown
            category_text = ""
            for category, stats in results["category_stats"].items():
                emoji_map = {
                    "common": "🏠",
                    "fusion": "🌍", 
                    "dietary": "🥗",
                    "technique": "🔧",
                    "edge_case": "🔥"
                }
                emoji = emoji_map.get(category, "📋")
                category_text += f"{emoji} {category.title()}: {stats['success_rate']:.1%} ({stats['successful']}/{stats['total']})\n"
            
            embed["fields"].append({
                "name": "🎯 Category Performance",
                "value": category_text.strip(),
                "inline": True
            })
            
            # Show top performing recipes with their best modes
            top_recipes = sorted(results["individual_results"], key=lambda x: x["overall_quality"], reverse=True)[:5]
            top_text = "\n".join([f"🏆 {r['name']}: {r['overall_quality']:.3f} ({r.get('best_mode', 'normal')})" for r in top_recipes])
            
            embed["fields"].append({
                "name": "🏆 Top Recipes",
                "value": top_text,
                "inline": True
            })
            
            # Add mode performance breakdown
            mode_stats = {"greedy": 0, "normal": 0, "creative": 0}
            total_recipes = len(results["individual_results"])
            
            for result in results["individual_results"]:
                best_mode = result.get("best_mode", "normal")
                mode_stats[best_mode] += 1
            
            mode_text = "\n".join([f"🎯 {mode.title()}: {count}/{total_recipes} ({count/total_recipes:.1%})" 
                                  for mode, count in mode_stats.items()])
            
            embed["fields"].append({
                "name": "🎯 Best Mode Distribution",
                "value": mode_text,
                "inline": True
            })
            
            # Show any problematic recipes
            problem_recipes = [r for r in results["individual_results"] if r["status"] != "SUCCESS"]
            if problem_recipes:
                problem_text = "\n".join([f"⚠️ {r['name']}: {r['overall_quality']:.3f}" for r in problem_recipes[:3]])
                embed["fields"].append({
                    "name": "⚠️ Needs Improvement",
                    "value": problem_text,
                    "inline": True
                })
            
            payload = {
                "embeds": [embed],
                "username": "Chef Genius Recipe Generator"
            }
            
            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
            print(f"\n🔔 Discord notification sent successfully!")
            
        except Exception as e:
            print(f"⚠️ Failed to send Discord notification: {e}")
    
    def save_detailed_results(self, results: Dict[str, Any], output_file: str = None):
        """Save detailed results to file."""
        
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = os.path.basename(results["checkpoint_path"]).replace("/", "_")
            output_file = f"recipe_generation_results_{checkpoint_name}_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Detailed results saved to: {output_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")

def main():
    """Main function for recipe generation testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Recipe Generation Testing Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path to test')
    parser.add_argument('--discord-webhook', type=str, help='Discord webhook URL for notifications')
    parser.add_argument('--parallel', type=int, default=2, help='Number of parallel generations')
    parser.add_argument('--output', type=str, help='Output file for detailed results')
    parser.add_argument('--max-length', type=int, default=600, help='Maximum generation length')
    
    args = parser.parse_args()
    
    print("🍳 CHEF GENIUS RECIPE GENERATION TESTER")
    print("🧪 Generating 20 diverse recipes from common to edge cases")
    print("=" * 80)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Create generator
    generator = RecipeGenerator(discord_webhook=args.discord_webhook)
    
    # Generate all recipes
    results = generator.generate_all_recipes(args.checkpoint, parallel_generations=args.parallel)
    
    if "error" not in results:
        # Print summary
        print(f"\n🎉 RECIPE GENERATION COMPLETE!")
        print(f"   📊 Success Rate: {results['success_rate']:.1%}")
        print(f"   🎯 Average Quality: {results['avg_quality_score']:.3f}")
        print(f"   ⚡ Performance: {results['avg_tokens_per_second']:.1f} tokens/sec")
        
        # Send Discord notification
        generator.send_discord_notification(results)
        
        # Save detailed results
        generator.save_detailed_results(results, args.output)
        
        # Print category summary
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, stats in results["category_stats"].items():
            print(f"   {category.title()}: {stats['success_rate']:.1%} success, {stats['avg_quality']:.3f} quality")
        
    else:
        print(f"❌ Generation failed: {results['error']}")

if __name__ == "__main__":
    main()