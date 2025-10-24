#!/usr/bin/env python3
"""
Base Recipe Generator
Shared base class for all specialized recipe generators with common functionality
"""

import os
import sys
import time
import torch
import requests
import json
import traceback
import re
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from tqdm import tqdm
import yaml

@dataclass
class RecipePrompt:
    """Base recipe generation prompt."""
    name: str
    prompt: str
    category: str
    difficulty: str = "normal"
    expected_features: List[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecipePrompt':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class GenerationConfig:
    """Configuration for recipe generation."""
    max_length: int = 600
    generation_modes: List[str] = None
    enable_beam_search: bool = False
    beam_size: int = 3
    num_return_sequences: int = 1
    enable_few_shot: bool = True
    retry_on_failure: bool = True
    max_retries: int = 2
    quality_threshold: float = 0.6

    def __post_init__(self):
        if self.generation_modes is None:
            self.generation_modes = ["greedy", "normal", "creative"]


@dataclass
class ValidationResult:
    """Result of recipe validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    cooking_time_valid: bool
    temperature_valid: bool
    quantity_valid: bool
    ingredient_instruction_aligned: bool


class BaseRecipeGenerator(ABC):
    """Base class for all recipe generators with shared functionality."""

    # Few-shot examples for better generation quality
    FEW_SHOT_EXAMPLES = """
Example Recipe 1:

RECIPE: Classic Roasted Chicken
INGREDIENTS:
- 1 whole chicken (4-5 lbs)
- 2 tablespoons olive oil
- 2 teaspoons salt
- 1 teaspoon black pepper
- 4 cloves garlic, minced
- 1 lemon, halved

INSTRUCTIONS:
1. Preheat oven to 425°F.
2. Pat chicken dry with paper towels.
3. Rub chicken with olive oil, salt, and pepper.
4. Stuff cavity with garlic and lemon halves.
5. Place chicken in roasting pan.
6. Roast for 60-75 minutes until internal temperature reaches 165°F.
7. Rest for 10 minutes before carving.

NOTES:
- Use a meat thermometer for accuracy
- Let rest to redistribute juices
- Save drippings for gravy

Example Recipe 2:

RECIPE: Chocolate Chip Cookies
INGREDIENTS:
- 2 1/4 cups all-purpose flour
- 1 teaspoon baking soda
- 1 cup butter, softened
- 3/4 cup sugar
- 2 eggs
- 2 cups chocolate chips

INSTRUCTIONS:
1. Preheat oven to 375°F.
2. Mix flour and baking soda in bowl.
3. Cream butter and sugar until fluffy.
4. Beat in eggs one at a time.
5. Gradually stir in flour mixture.
6. Fold in chocolate chips.
7. Drop spoonfuls onto baking sheet.
8. Bake 9-11 minutes until golden.

NOTES:
- Don't overbake for chewy cookies
- Cool on baking sheet 2 minutes
- Store in airtight container

---
"""

    def __init__(self, discord_webhook: str = None, config: GenerationConfig = None):
        """Initialize base recipe generator."""
        self.discord_webhook = discord_webhook
        self.config = config or GenerationConfig()
        self.results = []
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generation_cache = {}  # Cache tokenized prompts

        # Create recipe prompts (to be implemented by subclasses)
        self.recipe_prompts = self._create_prompts()

    @abstractmethod
    def _create_prompts(self) -> List[RecipePrompt]:
        """Create recipe prompts - must be implemented by subclasses."""
        pass

    @abstractmethod
    def get_recipe_type(self) -> str:
        """Get the type of recipes this generator creates."""
        pass

    def load_prompts_from_file(self, filepath: str) -> List[RecipePrompt]:
        """Load prompts from JSON or YAML file."""
        path = Path(filepath)

        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {filepath}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif path.suffix == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        prompts = []
        for item in data.get('prompts', []):
            prompts.append(RecipePrompt.from_dict(item))

        return prompts

    def save_prompts_to_file(self, filepath: str, prompts: List[RecipePrompt] = None):
        """Save prompts to JSON or YAML file."""
        path = Path(filepath)
        prompts = prompts or self.recipe_prompts

        data = {'prompts': [p.to_dict() for p in prompts]}

        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix in ['.yaml', '.yml']:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
            elif path.suffix == '.json':
                json.dump(data, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported file format: {path.suffix}")

        print(f"✅ Saved {len(prompts)} prompts to {filepath}")

    def load_model_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model and tokenizer from checkpoint with auto-detection."""
        try:
            print(f"🔄 Loading model from: {checkpoint_path}")

            if checkpoint_path.endswith('config.json'):
                checkpoint_path = os.path.dirname(checkpoint_path)

            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_path}")

            required_files = ['config.json']
            missing_files = [f for f in required_files if not os.path.exists(os.path.join(checkpoint_path, f))]
            if missing_files:
                raise FileNotFoundError(f"Missing files in checkpoint: {missing_files}")

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, local_files_only=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            # Auto-detect model type
            config = AutoConfig.from_pretrained(checkpoint_path, local_files_only=True)
            print(f"📋 Detected model type: {config.model_type}")

            # Load model based on type
            if config.model_type in ['t5', 'mt5', 'bart', 'pegasus', 'mbart']:
                print(f"🔄 Loading as Seq2Seq model...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else None,
                    local_files_only=True
                )
                self.model_type = "seq2seq"
            else:
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

            self.model.eval()

            print(f"✅ Model loaded successfully")
            print(f"   Device: {self.device}")
            print(f"   Model type: {type(self.model).__name__} ({self.model_type})")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

            return True

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            return False

    def _build_generation_params(self, mode: str, use_beam_search: bool = False) -> Dict[str, Any]:
        """Build generation parameters based on mode."""
        if use_beam_search:
            return {
                "num_beams": self.config.beam_size,
                "num_return_sequences": self.config.num_return_sequences,
                "early_stopping": True,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "length_penalty": 1.0
            }

        if mode == "greedy":
            return {
                "do_sample": False,
                "repetition_penalty": 1.1,
                "no_repeat_ngram_size": 3,
                "early_stopping": True,
                "length_penalty": 1.0
            }
        elif mode == "normal":
            return {
                "do_sample": True,
                "temperature": 0.5,
                "top_p": 0.85,
                "top_k": 35,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 4,
                "early_stopping": True,
                "length_penalty": 1.1
            }
        elif mode == "creative":
            return {
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
            raise ValueError(f"Unknown generation mode: {mode}")

    def _format_prompt(self, prompt: RecipePrompt, include_few_shot: bool = True) -> str:
        """Format the prompt for generation."""
        if self.model_type == "seq2seq":
            base_prompt = f"""Write a complete {self.get_recipe_type()} recipe following this exact format:

RECIPE: {prompt.name}

INGREDIENTS:
- [List each ingredient with exact measurements]

INSTRUCTIONS:
1. [Step by step cooking instructions]

NOTES:
- [Any cooking tips or variations]

{prompt.prompt}

"""
            if include_few_shot and self.config.enable_few_shot:
                return self.FEW_SHOT_EXAMPLES + base_prompt + "Write the recipe now:"
            else:
                return base_prompt + "Write the recipe now:"
        else:
            # Causal LM
            base_prompt = f"""{self.get_recipe_type().upper()} RECIPE: {prompt.name}

INGREDIENTS:
- """
            if include_few_shot and self.config.enable_few_shot:
                return self.FEW_SHOT_EXAMPLES + base_prompt
            else:
                return base_prompt

    def generate_recipe(self, prompt: RecipePrompt, max_length: int = None,
                       generation_mode: str = "normal", attempt: int = 1) -> Dict[str, Any]:
        """Generate a recipe with detailed metrics."""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}

        max_length = max_length or self.config.max_length
        start_time = time.time()

        # Check cache for tokenized prompt
        cache_key = f"{prompt.name}_{generation_mode}_{attempt}"

        try:
            formatted_prompt = self._format_prompt(prompt, include_few_shot=True)

            if cache_key not in self.generation_cache:
                if self.model_type == "seq2seq":
                    inputs = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=512,
                        padding=True
                    ).to(self.device)
                else:
                    inputs = self.tokenizer(
                        formatted_prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=256
                    ).to(self.device)

                self.generation_cache[cache_key] = inputs
            else:
                inputs = self.generation_cache[cache_key]

            # Build generation kwargs
            use_beam = self.config.enable_beam_search and attempt > 1
            gen_params = self._build_generation_params(generation_mode, use_beam_search=use_beam)

            generation_kwargs = {
                "max_new_tokens": max_length + 250,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "output_scores": True,
                "return_dict_in_generate": True,
            }
            generation_kwargs.update({k: v for k, v in gen_params.items() if v is not None})

            # Generate with autocast for efficiency
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                outputs = self.model.generate(**inputs, **generation_kwargs)

            # Calculate perplexity
            perplexity = self._calculate_perplexity(outputs.scores) if hasattr(outputs, 'scores') else None

            # Decode output
            if self.model_type == "seq2seq":
                generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                recipe = generated_text.strip()
            else:
                generated_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
                formatted_prompt_text = self._format_prompt(prompt, include_few_shot=True)
                recipe = generated_text[len(formatted_prompt_text):].strip()
                if not recipe:
                    recipe = generated_text.strip()

            # Post-process recipe
            recipe = self._post_process_recipe(recipe, prompt)

            generation_time = time.time() - start_time

            # Calculate metrics
            word_count = len(recipe.split())
            char_count = len(recipe)
            sentence_count = len([s for s in recipe.split('.') if s.strip()])
            tokens_generated = len(outputs.sequences[0]) - len(inputs.input_ids[0])

            return {
                "recipe": recipe,
                "generation_time": generation_time,
                "word_count": word_count,
                "char_count": char_count,
                "sentence_count": sentence_count,
                "tokens_generated": max(0, tokens_generated),
                "tokens_per_second": max(0, tokens_generated) / generation_time if generation_time > 0 else 0,
                "perplexity": perplexity,
                "attempt": attempt,
                "mode": generation_mode,
                "used_beam_search": use_beam
            }

        except Exception as e:
            return {
                "error": str(e),
                "generation_time": time.time() - start_time,
                "attempt": attempt
            }

    def _calculate_perplexity(self, scores: Tuple) -> float:
        """Calculate perplexity from generation scores."""
        try:
            import torch.nn.functional as F
            log_probs = []
            for score in scores:
                probs = F.softmax(score, dim=-1)
                max_prob = torch.max(probs)
                log_probs.append(torch.log(max_prob))

            avg_log_prob = torch.mean(torch.stack(log_probs))
            perplexity = torch.exp(-avg_log_prob).item()
            return perplexity
        except Exception:
            return None

    def _post_process_recipe(self, recipe: str, prompt: RecipePrompt) -> str:
        """Post-process recipe - can be overridden by subclasses."""
        recipe = self._clean_recipe_text(recipe)
        recipe = self._structure_recipe(recipe, prompt)
        return recipe

    def _clean_recipe_text(self, recipe: str) -> str:
        """Clean up formatting issues and artifacts."""
        # Remove formatting artifacts
        recipe = re.sub(r'\*\*[^*]*\*\*', '', recipe)
        recipe = re.sub(r'Products?: \[.*?\]', '', recipe)
        recipe = re.sub(r'Requirements?:.*?(?=INGREDIENTS|INSTRUCTIONS|\n\n)', '', recipe, flags=re.DOTALL)
        recipe = re.sub(r'Step-by-step.*?(?=INGREDIENTS|INSTRUCTIONS|\n\n)', '', recipe, flags=re.DOTALL)
        recipe = re.sub(r'Instruments?:', 'INSTRUCTIONS:', recipe)
        recipe = re.sub(r'---+', '', recipe)
        recipe = re.sub(r'_{3,}', '', recipe)
        recipe = re.sub(r'\#+', '', recipe)
        recipe = re.sub(r'\s*\[\s*.*?\s*\]\s*', ' ', recipe)
        recipe = re.sub(r'\s{3,}', ' ', recipe)
        recipe = re.sub(r'\n{3,}', '\n\n', recipe)

        return recipe.strip()

    def _structure_recipe(self, recipe: str, prompt: RecipePrompt) -> str:
        """Structure the recipe into proper sections."""
        lines = recipe.split('\n')
        ingredients = []
        instructions = []
        notes = []
        current_section = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            line_lower = line.lower()

            # Detect section headers
            if any(header in line_lower for header in ['ingredients:', 'ingredient']):
                current_section = 'ingredients'
                continue
            elif any(header in line_lower for header in ['instructions:', 'directions:', 'steps:', 'method']):
                current_section = 'instructions'
                continue
            elif any(header in line_lower for header in ['notes:', 'tips:', 'serving']):
                current_section = 'notes'
                continue

            # Skip junk lines
            if any(junk in line_lower for junk in ['recipe:', prompt.name.lower(), 'write the recipe']):
                continue

            # Categorize content
            if current_section == 'ingredients':
                if self._looks_like_ingredient(line):
                    ingredients.append(self._clean_ingredient_line(line))
            elif current_section == 'instructions':
                if self._looks_like_instruction(line):
                    instructions.append(self._clean_instruction_line(line))
            elif current_section == 'notes':
                notes.append(line)
            else:
                # Auto-detect
                if self._looks_like_ingredient(line):
                    ingredients.append(self._clean_ingredient_line(line))
                elif self._looks_like_instruction(line):
                    instructions.append(self._clean_instruction_line(line))

        # Build structured recipe
        result = f"{prompt.name}\n{'=' * len(prompt.name)}\n\n"

        if ingredients:
            result += "INGREDIENTS:\n"
            for ing in ingredients[:20]:
                if not ing.startswith('-'):
                    ing = f"- {ing}"
                result += f"{ing}\n"
            result += "\n"

        if instructions:
            result += "INSTRUCTIONS:\n"
            for i, inst in enumerate(instructions[:15], 1):
                if not inst.startswith(f"{i}."):
                    inst = f"{i}. {inst}"
                result += f"{inst}\n"
            result += "\n"

        if notes:
            result += "NOTES:\n"
            for note in notes[:5]:
                if not note.startswith('-'):
                    note = f"- {note}"
                result += f"{note}\n"

        return result

    def _looks_like_ingredient(self, line: str) -> bool:
        """Check if line looks like an ingredient."""
        line_lower = line.lower()

        if len(line) > 100:
            return False

        return any(indicator in line_lower for indicator in [
            'cup', 'tablespoon', 'teaspoon', 'tsp', 'tbsp', 'oz', 'lb', 'gram', 'clove', 'pinch',
            'salt', 'pepper', 'garlic', 'onion', 'oil'
        ]) or line.startswith(('- ', '• ', '* ')) or any(char.isdigit() for char in line[:10])

    def _looks_like_instruction(self, line: str) -> bool:
        """Check if line looks like an instruction."""
        line_lower = line.lower()

        return any(verb in line_lower for verb in [
            'cook', 'heat', 'sear', 'roast', 'braise', 'season', 'place', 'remove', 'add', 'mix'
        ]) or line_lower.startswith(('1.', '2.', '3.', 'first', 'then', 'next', 'finally'))

    def _clean_ingredient_line(self, line: str) -> str:
        """Clean up an ingredient line."""
        line = line.strip()
        line = re.sub(r'^\d+\.\s*', '', line)
        if not line.startswith(('- ', '• ', '* ')) and len(line) > 0:
            line = f"- {line}"
        return line

    def _clean_instruction_line(self, line: str) -> str:
        """Clean up an instruction line."""
        line = line.strip()
        line = re.sub(r'^\d+\.\s*', '', line)
        return line

    def validate_recipe(self, recipe: str, prompt: RecipePrompt) -> ValidationResult:
        """Validate recipe content for correctness."""
        errors = []
        warnings = []

        recipe_lower = recipe.lower()

        # Extract ingredients
        ingredients = []
        instructions_text = ""

        in_ingredients = False
        in_instructions = False

        for line in recipe.split('\n'):
            line_clean = line.strip()
            line_lower = line_clean.lower()

            if 'ingredients:' in line_lower:
                in_ingredients = True
                in_instructions = False
                continue
            elif 'instructions:' in line_lower or 'directions:' in line_lower:
                in_ingredients = False
                in_instructions = True
                continue
            elif 'notes:' in line_lower:
                in_ingredients = False
                in_instructions = False
                continue

            if in_ingredients and line_clean.startswith('-'):
                ingredients.append(line_clean[1:].strip().lower())
            elif in_instructions:
                instructions_text += " " + line_clean.lower()

        # Check cooking times
        cooking_time_valid = True
        time_pattern = r'(\d+)\s*(hour|hr|minute|min)'
        times = re.findall(time_pattern, recipe_lower)

        for time_val, unit in times:
            time_num = int(time_val)
            if unit in ['hour', 'hr'] and time_num > 24:
                warnings.append(f"Unusually long cooking time: {time_val} {unit}")
                cooking_time_valid = False
            elif unit in ['minute', 'min'] and time_num > 1440:  # 24 hours
                errors.append(f"Invalid cooking time: {time_val} {unit}")
                cooking_time_valid = False

        # Check temperatures
        temperature_valid = True
        temp_pattern = r'(\d+)\s*°?[FCfc]'
        temps = re.findall(temp_pattern, recipe_lower)

        for temp in temps:
            temp_num = int(temp)
            # Assume Fahrenheit
            if temp_num < 32 or temp_num > 600:
                errors.append(f"Invalid temperature: {temp}°F")
                temperature_valid = False
            elif temp_num > 500:
                warnings.append(f"Very high temperature: {temp}°F")

        # Check ingredient quantities
        quantity_valid = True
        quantity_pattern = r'(\d+)\s*(cup|tablespoon|teaspoon|lb|oz)'
        quantities = re.findall(quantity_pattern, recipe_lower)

        for qty, unit in quantities:
            qty_num = int(qty)
            if unit in ['cup'] and qty_num > 20:
                warnings.append(f"Large quantity: {qty} {unit}")
            elif unit in ['tablespoon', 'teaspoon'] and qty_num > 50:
                warnings.append(f"Large quantity: {qty} {unit}")
                quantity_valid = False

        # Check ingredient-instruction alignment
        ingredient_instruction_aligned = True
        main_ingredients = []

        for ing in ingredients:
            # Extract main ingredient name (first noun-like word after quantity)
            words = ing.split()
            for word in words:
                if word not in ['cup', 'tablespoon', 'teaspoon', 'tsp', 'tbsp', 'lb', 'oz', 'gram', 'of', 'the', 'a', 'an']:
                    if len(word) > 3:
                        main_ingredients.append(word)
                        break

        unused_ingredients = []
        for ing in main_ingredients:
            if ing not in instructions_text and len(ing) > 4:
                unused_ingredients.append(ing)

        if unused_ingredients and len(unused_ingredients) > len(main_ingredients) * 0.3:
            warnings.append(f"Some ingredients may not be used: {', '.join(unused_ingredients[:3])}")
            ingredient_instruction_aligned = False

        is_valid = len(errors) == 0 and cooking_time_valid and temperature_valid

        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            cooking_time_valid=cooking_time_valid,
            temperature_valid=temperature_valid,
            quantity_valid=quantity_valid,
            ingredient_instruction_aligned=ingredient_instruction_aligned
        )

    def analyze_recipe_quality(self, recipe: str, prompt: RecipePrompt) -> Dict[str, Any]:
        """Analyze recipe quality - can be extended by subclasses."""
        recipe_lower = recipe.lower()

        # Feature coverage
        features_found = []
        features_missing = []

        if prompt.expected_features:
            for feature in prompt.expected_features:
                if feature.lower() in recipe_lower:
                    features_found.append(feature)
                else:
                    features_missing.append(feature)

        feature_coverage = len(features_found) / len(prompt.expected_features) if prompt.expected_features else 1.0

        # Structure analysis
        has_ingredients = 'ingredients:' in recipe_lower
        has_instructions = any(word in recipe_lower for word in ['instructions:', 'directions:', 'steps:'])
        has_timing = any(word in recipe_lower for word in ['minutes', 'hours', 'until', 'preheat'])
        has_quantities = any(char.isdigit() for char in recipe)

        structure_score = sum([has_ingredients, has_instructions, has_timing, has_quantities]) / 4.0

        # Content quality
        word_count = len(recipe.split())
        is_appropriate_length = 100 <= word_count <= 1200

        # Overall quality
        quality_components = {
            "feature_coverage": feature_coverage * 0.30,
            "structure": structure_score * 0.30,
            "length": (1.0 if is_appropriate_length else 0.6) * 0.20,
            "timing": (1.0 if has_timing else 0.5) * 0.20,
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
            "is_appropriate_length": is_appropriate_length,
            "word_count": word_count,
            "quality_components": quality_components
        }

    def generate_single_recipe(self, prompt: RecipePrompt) -> Dict[str, Any]:
        """Generate and analyze a single recipe with retry logic."""
        print(f"🍳 Generating: {prompt.name} ({prompt.category.upper()})")

        start_time = time.time()
        generation_modes = self.config.generation_modes
        mode_results = {}

        for mode in generation_modes:
            print(f"  🎯 {mode.upper()} mode...")

            # Try generation with retry logic
            best_result = None
            best_quality = 0

            for attempt in range(1, self.config.max_retries + 1):
                mode_result = self.generate_recipe(prompt, generation_mode=mode, attempt=attempt)

                if "error" in mode_result:
                    mode_results[mode] = {
                        "status": "FAILED",
                        "error": mode_result["error"],
                        "generation_time": mode_result.get("generation_time", 0)
                    }
                    break

                # Analyze quality
                quality_result = self.analyze_recipe_quality(mode_result["recipe"], prompt)
                validation_result = self.validate_recipe(mode_result["recipe"], prompt)

                current_quality = quality_result["overall_quality"]

                if best_result is None or current_quality > best_quality:
                    best_result = mode_result
                    best_quality = current_quality

                # If quality is good enough, stop retrying
                if current_quality >= self.config.quality_threshold or not self.config.retry_on_failure:
                    break

                if attempt < self.config.max_retries:
                    print(f"    ⚠️ Quality {current_quality:.3f} below threshold, retrying...")

            if best_result:
                quality_result = self.analyze_recipe_quality(best_result["recipe"], prompt)
                validation_result = self.validate_recipe(best_result["recipe"], prompt)

                mode_results[mode] = {
                    "recipe": best_result["recipe"],
                    "status": "SUCCESS" if quality_result["overall_quality"] >= 0.4 else "NEEDS_IMPROVEMENT",
                    "generation_time": best_result["generation_time"],
                    "overall_quality": quality_result["overall_quality"],
                    "structure_score": quality_result["structure_score"],
                    "feature_coverage": quality_result["feature_coverage"],
                    "word_count": best_result["word_count"],
                    "tokens_per_second": best_result["tokens_per_second"],
                    "perplexity": best_result.get("perplexity"),
                    "features_found": quality_result["features_found"],
                    "features_missing": quality_result["features_missing"],
                    "quality_components": quality_result["quality_components"],
                    "validation": asdict(validation_result),
                    "attempts": best_result["attempt"]
                }

                status_emoji = "✅" if mode_results[mode]["status"] == "SUCCESS" else "⚠️"
                print(f"    {status_emoji} Quality: {quality_result['overall_quality']:.3f} (attempt {best_result['attempt']})")

                if not validation_result.is_valid:
                    print(f"    ⚠️ Validation errors: {', '.join(validation_result.errors)}")

        # Determine best mode
        successful_modes = [mode for mode, result in mode_results.items() if result.get("status") == "SUCCESS"]
        if successful_modes:
            best_mode = max(successful_modes, key=lambda m: mode_results[m]["overall_quality"])
        else:
            best_mode = max(mode_results.keys(), key=lambda m: mode_results[m].get("overall_quality", 0))

        total_time = time.time() - start_time

        result = {
            "name": prompt.name,
            "category": prompt.category,
            "difficulty": prompt.difficulty,
            "prompt": prompt.prompt,
            "status": mode_results[best_mode].get("status", "FAILED"),
            "total_time": total_time,
            "best_mode": best_mode,
            "mode_results": mode_results,
            "recipe": mode_results[best_mode].get("recipe", ""),
            "generation_time": sum(r.get("generation_time", 0) for r in mode_results.values()),
            "overall_quality": mode_results[best_mode].get("overall_quality", 0),
            "structure_score": mode_results[best_mode].get("structure_score", 0),
            "feature_coverage": mode_results[best_mode].get("feature_coverage", 0),
            "word_count": mode_results[best_mode].get("word_count", 0),
            "tokens_per_second": sum(r.get("tokens_per_second", 0) for r in mode_results.values() if r.get("tokens_per_second")) / len([r for r in mode_results.values() if r.get("tokens_per_second")]) if mode_results else 0,
            "perplexity": mode_results[best_mode].get("perplexity"),
            "features_found": mode_results[best_mode].get("features_found", []),
            "features_missing": mode_results[best_mode].get("features_missing", []),
            "quality_components": mode_results[best_mode].get("quality_components", {}),
            "validation": mode_results[best_mode].get("validation", {})
        }

        return result

    def generate_all_recipes(self, checkpoint_path: str, parallel_generations: int = 2) -> Dict[str, Any]:
        """Generate all recipes with progress tracking."""
        print(f"\n🍳 {self.get_recipe_type().upper()} RECIPE GENERATION SESSION")
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"🍳 Total recipes to generate: {len(self.recipe_prompts)}")
        print(f"⚡ Parallel generations: {parallel_generations}")
        print("=" * 80)

        if not self.load_model_from_checkpoint(checkpoint_path):
            return {"error": "Failed to load model"}

        start_time = time.time()
        results = []

        # Group by category
        categories = {}
        for prompt in self.recipe_prompts:
            if prompt.category not in categories:
                categories[prompt.category] = []
            categories[prompt.category].append(prompt)

        # Generate with progress bar
        with tqdm(total=len(self.recipe_prompts), desc="Generating recipes") as pbar:
            for category, prompts in categories.items():
                print(f"\n🎯 Generating {category.upper()} recipes ({len(prompts)} items)...")

                with ThreadPoolExecutor(max_workers=min(parallel_generations, 2)) as executor:
                    future_to_prompt = {executor.submit(self.generate_single_recipe, prompt): prompt for prompt in prompts}

                    for future in as_completed(future_to_prompt):
                        result = future.result()
                        results.append(result)
                        pbar.update(1)

                        # Clear some memory
                        if len(results) % 5 == 0:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()

                        status_emoji = "✅" if result["status"] == "SUCCESS" else "⚠️"
                        quality_emoji = "🏆" if result["overall_quality"] > 0.8 else "🎯" if result["overall_quality"] > 0.6 else "📈"
                        print(f"  {status_emoji} {result['name']}: {quality_emoji} {result['overall_quality']:.3f} quality ({result['best_mode']})")

        total_time = time.time() - start_time

        # Compile statistics
        successful_results = [r for r in results if r["status"] == "SUCCESS"]

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
            "recipe_type": self.get_recipe_type(),
            "checkpoint_path": checkpoint_path,
            "total_recipes": len(results),
            "successful_recipes": len(successful_results),
            "success_rate": len(successful_results) / len(results) if results else 0,
            "avg_quality_score": sum(r["overall_quality"] for r in results) / len(results) if results else 0,
            "avg_generation_time": sum(r["generation_time"] for r in results) / len(results) if results else 0,
            "avg_tokens_per_second": sum(r["tokens_per_second"] for r in results if r.get("tokens_per_second")) / len([r for r in results if r.get("tokens_per_second")]) if results else 0,
            "total_generation_time": total_time,
            "category_stats": category_stats,
            "individual_results": results,
            "config": asdict(self.config)
        }

        return comprehensive_results

    def send_discord_notification(self, results: Dict[str, Any]):
        """Send results to Discord - can be customized by subclasses."""
        if not self.discord_webhook:
            print("⚠️ No Discord webhook provided, skipping notification")
            return

        try:
            success_rate = results["success_rate"]
            avg_quality = results["avg_quality_score"]

            if success_rate >= 0.8 and avg_quality >= 0.7:
                color = 0x00ff00
                status_emoji = "✅"
                status_text = "EXCELLENT RESULTS"
            elif success_rate >= 0.6 and avg_quality >= 0.5:
                color = 0xffaa00
                status_emoji = "🎯"
                status_text = "GOOD RESULTS"
            else:
                color = 0xff0000
                status_emoji = "⚠️"
                status_text = "NEEDS IMPROVEMENT"

            embed = {
                "title": f"{status_emoji} {self.get_recipe_type().title()} Recipe Generation Results",
                "description": f"**{status_text}** - Generated {results['total_recipes']} recipes",
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
                category_text += f"📋 {category.title()}: {stats['success_rate']:.1%} ({stats['successful']}/{stats['total']})\n"

            embed["fields"].append({
                "name": "🎯 Category Performance",
                "value": category_text.strip(),
                "inline": True
            })

            # Top recipes
            top_recipes = sorted(results["individual_results"], key=lambda x: x["overall_quality"], reverse=True)[:5]
            top_text = "\n".join([f"🍳 {r['name']}: {r['overall_quality']:.3f}" for r in top_recipes])

            embed["fields"].append({
                "name": "🏆 Top Recipes",
                "value": top_text,
                "inline": True
            })

            payload = {
                "embeds": [embed],
                "username": f"Chef Genius {self.get_recipe_type().title()} Generator"
            }

            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()

            print(f"\n🔔 Discord notification sent successfully!")

        except Exception as e:
            print(f"⚠️ Failed to send Discord notification: {e}")

    def cleanup(self):
        """Clean up resources and cache."""
        self.generation_cache.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("🧹 Cleaned up resources")
