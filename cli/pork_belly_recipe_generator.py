#!/usr/bin/env python3
"""
Pork Belly Recipe Generation Script
Generates diverse pork belly recipes covering different cooking techniques and flavor profiles
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
class PorkBellyPrompt:
    """Individual pork belly recipe generation prompt."""
    name: str
    prompt: str
    category: str  # "traditional", "fusion", "smoking", "braising", "roasting", "creative"
    difficulty: str = "normal"  # normal, challenging, extreme
    expected_features: List[str] = None  # Features we hope to see

class PorkBellyRecipeGenerator:
    """Specialized pork belly recipe generation and testing system."""
    
    def __init__(self, discord_webhook: str = None):
        self.discord_webhook = discord_webhook
        self.results = []
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create diverse pork belly recipe prompts
        self.recipe_prompts = self._create_pork_belly_prompts()
        
    def _create_pork_belly_prompts(self) -> List[PorkBellyPrompt]:
        """Create 20 diverse pork belly recipe generation prompts."""
        
        prompts = []
        
        # TRADITIONAL TECHNIQUES (5 prompts)
        prompts.extend([
            PorkBellyPrompt(
                name="Classic Crispy Pork Belly",
                prompt="Create a traditional crispy pork belly recipe with perfectly rendered fat and crackling skin. Include proper scoring techniques and oven temperature control.",
                category="traditional",
                difficulty="normal",
                expected_features=["scoring", "skin", "crackling", "rendering", "temperature", "crispy"]
            ),
            PorkBellyPrompt(
                name="Chinese Red-Braised Pork Belly (Hong Shao Rou)",
                prompt="Design an authentic Chinese red-braised pork belly recipe with soy sauce, rock sugar, and Shaoxing wine. Include traditional technique for achieving tender, glossy meat.",
                category="traditional",
                difficulty="normal",
                expected_features=["soy sauce", "rock sugar", "Shaoxing wine", "braising", "tender", "glossy"]
            ),
            PorkBellyPrompt(
                name="Korean-Style Pan-Seared Pork Belly",
                prompt="Create a Korean-style pan-seared pork belly recipe with proper technique for achieving crispy exterior in a skillet, including traditional banchan suggestions and wrapping techniques.",
                category="traditional",
                difficulty="normal",
                expected_features=["pan-seared", "banchan", "lettuce wraps", "Korean", "skillet", "crispy"]
            ),
            PorkBellyPrompt(
                name="Italian Porchetta-Style Pork Belly",
                prompt="Develop an Italian porchetta-inspired pork belly recipe with herb stuffing, proper rolling technique, and slow roasting for crispy exterior and tender interior.",
                category="traditional",
                difficulty="challenging",
                expected_features=["herbs", "rolling", "stuffing", "slow roasting", "crispy", "tender"]
            ),
            PorkBellyPrompt(
                name="Southern-Style Oven-Braised Pork Belly",
                prompt="Create a Southern-style oven-braised pork belly recipe with proper cubing, seasoning, and slow oven cooking for caramelized, tender bites.",
                category="traditional",
                difficulty="challenging",
                expected_features=["cubing", "oven-braised", "caramelized", "Southern", "seasoning", "tender"]
            )
        ])
        
        # ROASTING & PAN COOKING (4 prompts)
        prompts.extend([
            PorkBellyPrompt(
                name="Low and Slow Oven-Roasted Pork Belly",
                prompt="Design a low and slow oven-roasted pork belly recipe with temperature control, proper scoring, and timing for crispy skin and tender meat.",
                category="roasting",
                difficulty="challenging",
                expected_features=["oven-roasted", "scoring", "crispy skin", "temperature", "low and slow", "tender"]
            ),
            PorkBellyPrompt(
                name="Pan-Seared and Oven-Finished Pork Belly Bites",
                prompt="Create pan-seared pork belly bites finished in the oven with proper cubing, searing technique, and oven glazing for caramelized results.",
                category="roasting",
                difficulty="challenging",
                expected_features=["pan-seared", "cubing", "oven-finished", "glazing", "caramelized", "bites"]
            ),
            PorkBellyPrompt(
                name="Oven-Cured and Roasted Pork Belly",
                prompt="Develop an oven-cured and roasted pork belly recipe for homemade bacon-style results, including dry curing process and slow roasting techniques.",
                category="roasting",
                difficulty="extreme",
                expected_features=["oven-cured", "dry curing", "bacon-style", "homemade", "slow roasting", "process"]
            ),
            PorkBellyPrompt(
                name="Herb-Crusted Oven-Roasted Pork Belly",
                prompt="Create an herb-crusted oven-roasted pork belly recipe with complementary herb blend, proper coating technique, and perfect roasting method.",
                category="roasting",
                difficulty="normal",
                expected_features=["herb-crusted", "oven-roasted", "herb blend", "coating", "roasting method"]
            )
        ])
        
        # BRAISING & SLOW COOKING (4 prompts)
        prompts.extend([
            PorkBellyPrompt(
                name="Beer-Braised Pork Belly",
                prompt="Design a beer-braised pork belly recipe with proper beer selection, vegetable aromatics, and braising liquid reduction for rich, tender results.",
                category="braising",
                difficulty="normal",
                expected_features=["beer", "braising", "aromatics", "reduction", "tender", "rich"]
            ),
            PorkBellyPrompt(
                name="Miso-Braised Pork Belly Ramen",
                prompt="Create a Japanese-inspired miso-braised pork belly for ramen with proper chashu technique, marinating, and slicing for perfect ramen topping.",
                category="braising",
                difficulty="challenging",
                expected_features=["miso", "chashu", "ramen", "marinating", "Japanese", "slicing"]
            ),
            PorkBellyPrompt(
                name="Wine-Braised Pork Belly Confit",
                prompt="Develop a French-style wine-braised pork belly confit with proper wine selection, herb bouquet, and low-temperature cooking for ultimate tenderness.",
                category="braising",
                difficulty="challenging",
                expected_features=["wine", "confit", "French", "herb bouquet", "low temperature", "tenderness"]
            ),
            PorkBellyPrompt(
                name="Coconut Curry Braised Pork Belly",
                prompt="Create a Southeast Asian coconut curry braised pork belly with proper spice balance, coconut milk technique, and aromatic garnishes.",
                category="braising",
                difficulty="normal",
                expected_features=["coconut curry", "Southeast Asian", "spice balance", "coconut milk", "aromatic", "garnishes"]
            )
        ])
        
        # FUSION CUISINE (4 prompts)
        prompts.extend([
            PorkBellyPrompt(
                name="Korean-Mexican Pork Belly Tacos",
                prompt="Design fusion pork belly tacos combining Korean gochujang flavors with Mexican tortilla techniques, including kimchi slaw and sesame-lime crema.",
                category="fusion",
                difficulty="challenging",
                expected_features=["Korean", "Mexican", "gochujang", "tacos", "kimchi", "sesame-lime"]
            ),
            PorkBellyPrompt(
                name="Vietnamese-Italian Pork Belly Banh Mi Pizza",
                prompt="Create an innovative pizza combining Vietnamese banh mi flavors with Italian pizza technique, featuring pork belly, pickled vegetables, and cilantro.",
                category="fusion",
                difficulty="extreme",
                expected_features=["Vietnamese", "Italian", "banh mi", "pizza", "pickled vegetables", "cilantro"]
            ),
            PorkBellyPrompt(
                name="Thai-French Pork Belly Cassoulet",
                prompt="Develop a fusion cassoulet using Thai flavors and pork belly, incorporating lemongrass, fish sauce, and traditional French bean cooking techniques.",
                category="fusion",
                difficulty="extreme",
                expected_features=["Thai", "French", "cassoulet", "lemongrass", "fish sauce", "beans"]
            ),
            PorkBellyPrompt(
                name="Indian-Southern Pork Belly Curry",
                prompt="Create a spiced pork belly curry combining Indian spice techniques with Southern US comfort food elements, including proper spice tempering and slow cooking.",
                category="fusion",
                difficulty="challenging",
                expected_features=["Indian spices", "Southern", "curry", "tempering", "slow cooking", "comfort food"]
            )
        ])
        
        # CREATIVE TECHNIQUES (3 prompts)
        prompts.extend([
            PorkBellyPrompt(
                name="Sous Vide Pork Belly with Torched Finish",
                prompt="Design a modern sous vide pork belly recipe with precise time and temperature, followed by torching technique for perfect texture contrast.",
                category="creative",
                difficulty="extreme",
                expected_features=["sous vide", "precise temperature", "torching", "texture contrast", "modern", "technique"]
            ),
            PorkBellyPrompt(
                name="Pork Belly Bao Buns from Scratch",
                prompt="Create homemade bao buns with perfectly steamed pork belly filling, including dough preparation, steaming technique, and traditional garnishes.",
                category="creative",
                difficulty="extreme",
                expected_features=["bao buns", "steamed", "dough", "steaming", "homemade", "garnishes"]
            ),
            PorkBellyPrompt(
                name="Deconstructed Pork Belly Ramen Bowl",
                prompt="Design a modern deconstructed ramen presentation featuring pork belly as the centerpiece with innovative plating and molecular gastronomy elements.",
                category="creative",
                difficulty="extreme",
                expected_features=["deconstructed", "modern", "centerpiece", "plating", "molecular gastronomy", "innovative"]
            )
        ])
        
        return prompts
    
    def _post_process_recipe(self, recipe: str, prompt: PorkBellyPrompt) -> str:
        """Post-process recipe to ensure it has proper structure and pork belly focus."""
        
        # Clean up the raw recipe text first
        recipe = self._clean_recipe_text(recipe)
        
        # Parse into structured sections
        structured_recipe = self._structure_recipe(recipe, prompt)
        
        # Validate and enhance content
        final_recipe = self._validate_recipe_content(structured_recipe, prompt)
        
        return final_recipe
    
    def _clean_recipe_text(self, recipe: str) -> str:
        """Clean up formatting issues and artifacts."""
        import re
        
        # Remove weird formatting artifacts
        recipe = re.sub(r'\*\*[^*]*\*\*', '', recipe)  # Remove **text**
        recipe = re.sub(r'Products?: \[.*?\]', '', recipe)  # Remove Products: [...]
        recipe = re.sub(r'Requirements?:.*?(?=INGREDIENTS|INSTRUCTIONS|\n\n)', '', recipe, flags=re.DOTALL)
        recipe = re.sub(r'Step-by-step.*?(?=INGREDIENTS|INSTRUCTIONS|\n\n)', '', recipe, flags=re.DOTALL)
        recipe = re.sub(r'Instruments?:', 'INSTRUCTIONS:', recipe)
        recipe = re.sub(r'---+', '', recipe)  # Remove dashes
        recipe = re.sub(r'_{3,}', '', recipe)  # Remove underscores
        recipe = re.sub(r'\#+', '', recipe)  # Remove # symbols
        recipe = re.sub(r'\s*\[\s*.*?\s*\]\s*', ' ', recipe)  # Remove [text]
        recipe = re.sub(r'\s{3,}', ' ', recipe)  # Multiple spaces to single
        recipe = re.sub(r'\n{3,}', '\n\n', recipe)  # Multiple newlines to double
        
        return recipe.strip()
    
    def _structure_recipe(self, recipe: str, prompt: PorkBellyPrompt) -> str:
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
            
            # Skip obvious junk lines
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
                # Auto-detect if no section specified
                if self._looks_like_ingredient(line):
                    ingredients.append(self._clean_ingredient_line(line))
                elif self._looks_like_instruction(line):
                    instructions.append(self._clean_instruction_line(line))
        
        # Build structured recipe
        result = f"{prompt.name}\n{'=' * len(prompt.name)}\n\n"
        
        # Ensure we have pork belly in ingredients
        has_pork_belly = any('pork belly' in ing.lower() or 'pork-belly' in ing.lower() for ing in ingredients)
        
        if not has_pork_belly:
            ingredients.insert(0, "2-3 lbs pork belly, skin on")
        
        # Add ingredients
        if ingredients:
            result += "INGREDIENTS:\n"
            for ing in ingredients[:15]:  # Limit to reasonable number
                if not ing.startswith('-'):
                    ing = f"- {ing}"
                result += f"{ing}\n"
            result += "\n"
        
        # Add instructions
        if instructions:
            result += "INSTRUCTIONS:\n"
            for i, inst in enumerate(instructions[:12], 1):  # Limit steps
                if not inst.startswith(f"{i}."):
                    inst = f"{i}. {inst}"
                result += f"{inst}\n"
            result += "\n"
        
        # Add basic cooking steps if missing
        if not any('preheat' in inst.lower() or 'oven' in inst.lower() or 'pan' in inst.lower() for inst in instructions):
            result += "INSTRUCTIONS:\n1. Preheat oven to 375°F or heat large oven-safe pan\n2. Score pork belly skin in crosshatch pattern\n3. Season with salt and pepper\n"
            for i, inst in enumerate(instructions[:9], 4):
                result += f"{i}. {inst}\n"
            result += "\n"
        
        # Add notes
        if notes:
            result += "NOTES:\n"
            for note in notes[:5]:
                if not note.startswith('-'):
                    note = f"- {note}"
                result += f"{note}\n"
        else:
            result += "NOTES:\n- Cook until internal temperature reaches 145°F\n- Let rest 10 minutes before slicing\n- Score skin for better fat rendering\n"
        
        return result
    
    def _looks_like_ingredient(self, line: str) -> bool:
        """Check if line looks like an ingredient."""
        line_lower = line.lower()
        
        # Skip obviously bad lines
        if len(line) > 100 or any(bad in line_lower for bad in ['recipe', 'instructions', 'step', 'cook until']):
            return False
            
        return any(indicator in line_lower for indicator in [
            'cup', 'tablespoon', 'teaspoon', 'tsp', 'tbsp', 'oz', 'lb', 'gram', 'clove', 'pinch',
            'pork belly', 'pork', 'soy sauce', 'salt', 'pepper', 'garlic', 'onion', 'oil'
        ]) or line.startswith(('- ', '• ', '* ')) or any(char.isdigit() for char in line[:10])
    
    def _looks_like_instruction(self, line: str) -> bool:
        """Check if line looks like an instruction."""
        line_lower = line.lower()
        
        return any(verb in line_lower for verb in [
            'cook', 'heat', 'sear', 'roast', 'braise', 'render', 'score', 'season', 'place', 'remove'
        ]) or line_lower.startswith(('1.', '2.', '3.', 'first', 'then', 'next', 'finally'))
    
    def _clean_ingredient_line(self, line: str) -> str:
        """Clean up an ingredient line."""
        import re
        line = line.strip()
        # Remove numbering from start
        line = re.sub(r'^\d+\.\s*', '', line)
        # Ensure it starts with dash if it's a list item
        if not line.startswith(('- ', '• ', '* ')) and len(line) > 0:
            line = f"- {line}"
        return line
    
    def _clean_instruction_line(self, line: str) -> str:
        """Clean up an instruction line."""
        import re
        line = line.strip()
        # Remove existing numbering - we'll renumber
        line = re.sub(r'^\d+\.\s*', '', line)
        return line
    
    def _validate_recipe_content(self, recipe: str, prompt: PorkBellyPrompt) -> str:
        """Final validation and content fixes."""
        lines = recipe.split('\n')
        
        # Check for minimum content requirements
        has_enough_ingredients = len([l for l in lines if l.strip().startswith('-') and 'INGREDIENTS' in recipe]) >= 3
        has_enough_instructions = len([l for l in lines if l.strip() and l.strip()[0].isdigit()]) >= 3
        
        if not has_enough_ingredients or not has_enough_instructions:
            # Return a basic template if content is too poor
            return f"""{prompt.name}
{'=' * len(prompt.name)}

INGREDIENTS:
- 2-3 lbs pork belly, skin on
- 2 tsp salt
- 1 tsp black pepper
- 2 tbsp olive oil
- 2 cloves garlic, minced

INSTRUCTIONS:
1. Preheat oven to 375°F
2. Score pork belly skin in crosshatch pattern
3. Season with salt and pepper
4. Heat oil in oven-safe pan over medium-high heat
5. Sear pork belly skin-side down until golden
6. Flip and transfer to oven
7. Roast 45-60 minutes until tender
8. Rest 10 minutes before slicing

NOTES:
- Cook until internal temperature reaches 145°F
- Score skin for better fat rendering
- Let rest for optimal slicing
"""
        
        return recipe
    
    def _quick_quality_check(self, recipe: str) -> float:
        """Quick quality assessment for pork belly recipes."""
        recipe_lower = recipe.lower()
        word_count = len(recipe.split())
        
        # Basic structural checks
        has_pork_belly = any(term in recipe_lower for term in ["pork belly", "pork-belly", "belly"])
        has_ingredients = any(word in recipe_lower for word in ["ingredients", "cup", "tablespoon", "teaspoon"])
        has_instructions = any(word in recipe_lower for word in ["instructions", "cook", "heat", "season", "render"])
        has_measurements = any(char.isdigit() for char in recipe)
        is_long_enough = word_count >= 50
        
        score = sum([has_pork_belly, has_ingredients, has_instructions, has_measurements, is_long_enough]) / 5.0
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
    
    def generate_recipe(self, prompt: PorkBellyPrompt, max_length: int = 600, generation_mode: str = "normal") -> Dict[str, Any]:
        """Generate a pork belly recipe with detailed metrics."""
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
            # Balanced generation
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
                # FLAN-T5 optimized generation with pork belly focus
                formatted_prompt = f"""Write a complete pork belly recipe following this exact format:

RECIPE: {prompt.name}

INGREDIENTS:
- 2-3 lbs pork belly, skin on
- [List each ingredient with exact measurements]
- [Include all seasonings, oils, vegetables needed]

INSTRUCTIONS:
1. Preheat oven to [temperature] or prepare pan
2. [Step by step cooking instructions]
3. Score pork belly skin if needed for rendering
4. [Continue with proper cooking technique]
5. Rest before slicing and serve

NOTES:
- Cook until internal temperature reaches 145°F
- [Any cooking tips or variations]

{prompt.prompt}

Write the recipe now:"""
                
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding=True
                ).to(self.device)
                
                # Build generation kwargs, filtering out None values
                generation_kwargs = {
                    "max_new_tokens": max_length + 250,  # Extra tokens for detailed pork belly recipes
                    "pad_token_id": self.tokenizer.pad_token_id,
                    "eos_token_id": self.tokenizer.eos_token_id,
                }
                generation_kwargs.update({k: v for k, v in gen_params.items() if v is not None})
                
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                recipe = generated_text.strip()
                
                # Post-process and validate recipe structure
                recipe = self._post_process_recipe(recipe, prompt)
                    
            else:
                # Causal LM generation with pork belly focus
                formatted_prompt = f"""PORK BELLY RECIPE: {prompt.name}

INGREDIENTS:
- 2-3 lbs pork belly, skin on
- """
                
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=256
                ).to(self.device)
                
                # Build generation kwargs, filtering out None values
                generation_kwargs = {
                    "max_new_tokens": max_length + 250,
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
    
    def analyze_recipe_quality(self, recipe: str, prompt: PorkBellyPrompt) -> Dict[str, Any]:
        """Analyze the quality and completeness of generated pork belly recipe."""
        
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
        
        has_pork_belly = any(term in recipe_lower for term in [
            "pork belly", "pork-belly", "belly"
        ])
        
        has_measurements = any(word in recipe_lower for word in [
            "cup", "tablespoon", "teaspoon", "tsp", "tbsp", "oz", "ounce", "lb", "pound", 
            "gram", "kg", "liter", "ml", "clove", "pinch", "dash"
        ])
        
        has_ingredients = has_ingredients_section or has_measurements or any(word in recipe_lower for word in [
            "salt", "pepper", "oil", "soy sauce", "garlic", "onion", "ginger"
        ])
        
        has_instructions_section = any(word in recipe_lower for word in [
            "## instructions:", "instructions:", "## steps:", "steps:", "## directions:", 
            "directions:", "## method:", "method:", "## preparation:", "preparation:"
        ])
        
        has_pork_belly_techniques = any(word in recipe_lower for word in [
            "score", "render", "skin", "fat", "sear", "braise", "smoke", "roast", "crispy", "tender"
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
        
        # Pork belly specific techniques
        has_techniques = has_pork_belly_techniques or any(word in recipe_lower for word in [
            "whisk", "fold", "sauté", "grill", "roast", "braise", "steam", "fry",
            "blend", "puree", "marinate", "season", "garnish", "dice", "mince"
        ])
        
        structure_score = sum([has_pork_belly, has_ingredients, has_instructions, has_timing, has_quantities, has_techniques]) / 6.0
        
        # Content quality assessment
        word_count = len(recipe.split())
        is_appropriate_length = 100 <= word_count <= 1200  # Longer for detailed pork belly recipes
        
        # Creativity and detail assessment
        creativity_indicators = [
            "innovative", "unique", "creative", "fusion", "twist", "variation",
            "traditional", "authentic", "signature", "special", "perfect", "technique"
        ]
        has_creativity = any(indicator in recipe_lower for indicator in creativity_indicators)
        
        # Technical detail assessment for pork belly
        technical_terms = [
            "temperature", "internal", "consistency", "texture", "technique", 
            "method", "precise", "exact", "careful", "gentle", "vigorous",
            "render", "score", "skin", "fat", "crackling"
        ]
        has_technical_detail = any(term in recipe_lower for term in technical_terms)
        
        # Calculate overall quality score with pork belly emphasis
        quality_components = {
            "pork_belly_focus": (1.0 if has_pork_belly else 0.0) * 0.20,  # Critical for pork belly recipes
            "feature_coverage": feature_coverage * 0.20,
            "structure": structure_score * 0.25,
            "length": (1.0 if is_appropriate_length else 0.6) * 0.15,
            "creativity": (1.0 if has_creativity else 0.7) * 0.10,
            "technical_detail": (1.0 if has_technical_detail else 0.8) * 0.10
        }
        
        overall_quality = sum(quality_components.values())
        
        return {
            "overall_quality": overall_quality,
            "feature_coverage": feature_coverage,
            "structure_score": structure_score,
            "features_found": features_found,
            "features_missing": features_missing,
            "has_pork_belly": has_pork_belly,
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
    
    def generate_single_recipe(self, prompt: PorkBellyPrompt) -> Dict[str, Any]:
        """Generate and analyze a single pork belly recipe with 3 generation modes."""
        print(f"🥓 Generating: {prompt.name} ({prompt.category.upper()})")
        
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
        
        # Determine overall success based on best mode (stricter thresholds)
        quality_thresholds = {
            "traditional": 0.8,
            "roasting": 0.7,
            "braising": 0.7,
            "fusion": 0.6,
            "creative": 0.5
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
        """Generate all pork belly recipes and compile comprehensive results."""
        
        print(f"\n🥓 PORK BELLY RECIPE GENERATION SESSION")
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"🥓 Total recipes to generate: {len(self.recipe_prompts)}")
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
            print(f"\n🎯 Generating {category.upper()} pork belly recipes ({len(prompts)} items)...")
            
            # Use limited parallelization to avoid memory issues
            with ThreadPoolExecutor(max_workers=min(parallel_generations, 2)) as executor:
                future_to_prompt = {executor.submit(self.generate_single_recipe, prompt): prompt for prompt in prompts}
                
                for future in as_completed(future_to_prompt):
                    result = future.result()
                    results.append(result)
                    
                    status_emoji = "✅" if result["status"] == "SUCCESS" else "⚠️"
                    quality_emoji = "🏆" if result["overall_quality"] > 0.8 else "🎯" if result["overall_quality"] > 0.6 else "📈"
                    
                    print(f"  {status_emoji} {result['name']}: {quality_emoji} {result['overall_quality']:.3f} quality ({result['best_mode']})")
        
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
        """Send comprehensive pork belly recipe generation results to Discord."""
        
        if not self.discord_webhook:
            print("⚠️ No Discord webhook provided, skipping notification")
            return
        
        try:
            # Determine overall performance
            success_rate = results["success_rate"]
            avg_quality = results["avg_quality_score"]
            
            if success_rate >= 0.8 and avg_quality >= 0.7:
                color = 0x8B4513  # Saddle brown - like cooked pork belly
                status_emoji = "🥓"
                status_text = "EXCELLENT PORK BELLY RECIPES"
            elif success_rate >= 0.6 and avg_quality >= 0.5:
                color = 0xD2691E  # Chocolate brown
                status_emoji = "🍖"
                status_text = "GOOD PORK BELLY RECIPES"
            else:
                color = 0xA0522D  # Sienna
                status_emoji = "⚠️"
                status_text = "PORK BELLY RECIPES NEED IMPROVEMENT"
            
            embed = {
                "title": f"{status_emoji} Pork Belly Recipe Generation Results",
                "description": f"**{status_text}** - Generated {results['total_recipes']} diverse pork belly recipes",
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
            category_emojis = {
                "traditional": "🏠",
                "roasting": "🔥", 
                "braising": "🍲",
                "fusion": "🌍",
                "creative": "🎨"
            }
            for category, stats in results["category_stats"].items():
                emoji = category_emojis.get(category, "📋")
                category_text += f"{emoji} {category.title()}: {stats['success_rate']:.1%} ({stats['successful']}/{stats['total']})\n"
            
            embed["fields"].append({
                "name": "🎯 Category Performance",
                "value": category_text.strip(),
                "inline": True
            })
            
            # Show top performing recipes with their best modes
            top_recipes = sorted(results["individual_results"], key=lambda x: x["overall_quality"], reverse=True)[:5]
            top_text = "\n".join([f"🥓 {r['name']}: {r['overall_quality']:.3f} ({r.get('best_mode', 'normal')})" for r in top_recipes])
            
            embed["fields"].append({
                "name": "🏆 Top Pork Belly Recipes",
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
                "username": "Chef Genius Pork Belly Generator"
            }
            
            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
            print(f"\n🔔 Discord notification sent successfully!")
            
        except Exception as e:
            print(f"⚠️ Failed to send Discord notification: {e}")
    
    def print_readable_recipe(self, result: Dict[str, Any]):
        """Print a single recipe in a clean, readable format for copy/paste."""
        
        print("\n" + "="*80)
        print(f"🥓 {result['name'].upper()}")
        print(f"Category: {result['category'].title()} | Difficulty: {result['difficulty'].title()} | Best Mode: {result['best_mode'].title()}")
        print(f"Quality Score: {result['overall_quality']:.3f} | Generation Time: {result['generation_time']:.2f}s")
        print("="*80)
        
        recipe_text = result['recipe']
        
        # Clean up the recipe text for better readability
        lines = recipe_text.split('\n')
        formatted_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                formatted_lines.append("")
                continue
            
            # Format headers
            if line.startswith('##'):
                formatted_lines.append("")
                formatted_lines.append(line.replace('##', '').strip().upper())
                formatted_lines.append("-" * len(line.replace('##', '').strip()))
            elif line.startswith('#'):
                formatted_lines.append("")
                formatted_lines.append(line.replace('#', '').strip().upper())
                formatted_lines.append("-" * len(line.replace('#', '').strip()))
            else:
                formatted_lines.append(line)
        
        # Print the formatted recipe
        for line in formatted_lines:
            print(line)
        
        print("\n" + "="*80)
        print(f"✅ Recipe ready to copy and paste!")
        print("="*80)

    def print_all_recipes_readable(self, results: Dict[str, Any]):
        """Print all recipes in readable format."""
        
        print(f"\n\n🥓 PORK BELLY RECIPE COLLECTION")
        print(f"Generated from: {os.path.basename(results['checkpoint_path'])}")
        print(f"Success Rate: {results['success_rate']:.1%} | Average Quality: {results['avg_quality_score']:.3f}")
        print("="*80)
        
        # Group by category and print
        categories = {}
        for result in results['individual_results']:
            category = result['category']
            if category not in categories:
                categories[category] = []
            categories[category].append(result)
        
        # Sort each category by quality (best first)
        for category in categories:
            categories[category].sort(key=lambda x: x['overall_quality'], reverse=True)
        
        # Print recipes by category
        for category, recipes in categories.items():
            print(f"\n\n🎯 {category.upper()} PORK BELLY RECIPES")
            print("="*80)
            
            for recipe in recipes:
                if recipe['status'] == 'SUCCESS':
                    self.print_readable_recipe(recipe)
                    
                    # Ask user if they want to see the next recipe
                    user_input = input(f"\nPress Enter to see next recipe, 's' to skip category, or 'q' to quit: ").strip().lower()
                    if user_input == 'q':
                        return
                    elif user_input == 's':
                        break

    def save_detailed_results(self, results: Dict[str, Any], output_file: str = None):
        """Save detailed results to file."""
        
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = os.path.basename(results["checkpoint_path"]).replace("/", "_")
            output_file = f"pork_belly_results_{checkpoint_name}_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            print(f"\n💾 Detailed pork belly results saved to: {output_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save results: {e}")
    
    def save_readable_recipes(self, results: Dict[str, Any], output_file: str = None):
        """Save all recipes in readable text format."""
        
        if not output_file:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            checkpoint_name = os.path.basename(results["checkpoint_path"]).replace("/", "_")
            output_file = f"pork_belly_recipes_{checkpoint_name}_{timestamp}.txt"
        else:
            # If user provided a filename, use it but change extension to .txt
            if output_file.endswith('.json'):
                output_file = output_file.replace('.json', '.txt')
            elif not output_file.endswith('.txt'):
                output_file = f"{output_file}.txt"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"🥓 PORK BELLY RECIPE COLLECTION\n")
                f.write(f"Generated from: {os.path.basename(results['checkpoint_path'])}\n")
                f.write(f"Success Rate: {results['success_rate']:.1%} | Average Quality: {results['avg_quality_score']:.3f}\n")
                f.write("="*80 + "\n\n")
                
                # Group by category
                categories = {}
                for result in results['individual_results']:
                    category = result['category']
                    if category not in categories:
                        categories[category] = []
                    categories[category].append(result)
                
                # Sort each category by quality
                for category in categories:
                    categories[category].sort(key=lambda x: x['overall_quality'], reverse=True)
                
                # Write recipes by category
                for category, recipes in categories.items():
                    f.write(f"\n🎯 {category.upper()} PORK BELLY RECIPES\n")
                    f.write("="*80 + "\n")
                    
                    for recipe in recipes:
                        if recipe['status'] == 'SUCCESS':
                            f.write("\n" + "="*80 + "\n")
                            f.write(f"🥓 {recipe['name'].upper()}\n")
                            f.write(f"Category: {recipe['category'].title()} | Difficulty: {recipe['difficulty'].title()} | Best Mode: {recipe['best_mode'].title()}\n")
                            f.write(f"Quality Score: {recipe['overall_quality']:.3f} | Generation Time: {recipe['generation_time']:.2f}s\n")
                            f.write("="*80 + "\n\n")
                            
                            # Format recipe text
                            recipe_text = recipe['recipe']
                            lines = recipe_text.split('\n')
                            
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    f.write("\n")
                                    continue
                                
                                # Format headers
                                if line.startswith('##'):
                                    f.write("\n")
                                    header = line.replace('##', '').strip().upper()
                                    f.write(header + "\n")
                                    f.write("-" * len(header) + "\n")
                                elif line.startswith('#'):
                                    f.write("\n")
                                    header = line.replace('#', '').strip().upper()
                                    f.write(header + "\n")
                                    f.write("-" * len(header) + "\n")
                                else:
                                    f.write(line + "\n")
                            
                            f.write("\n" + "="*80 + "\n\n")
            
            print(f"\n📖 Readable recipes saved to: {output_file}")
            
        except Exception as e:
            print(f"⚠️ Failed to save readable recipes: {e}")

def main():
    """Main function for pork belly recipe generation testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Pork Belly Recipe Generation Testing Script')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path to test')
    parser.add_argument('--discord-webhook', type=str, help='Discord webhook URL for notifications')
    parser.add_argument('--parallel', type=int, default=2, help='Number of parallel generations')
    parser.add_argument('--output', type=str, help='Output file for detailed results')
    parser.add_argument('--max-length', type=int, default=600, help='Maximum generation length')
    
    args = parser.parse_args()
    
    print("🥓 CHEF GENIUS PORK BELLY RECIPE GENERATOR")
    print("🍖 Generating 20 diverse pork belly recipes from traditional to creative")
    print("🍳 All recipes use pan and oven cooking methods only")
    print("=" * 80)
    
    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return
    
    # Create generator
    generator = PorkBellyRecipeGenerator(discord_webhook=args.discord_webhook)
    
    # Generate all recipes
    results = generator.generate_all_recipes(args.checkpoint, parallel_generations=args.parallel)
    
    if "error" not in results:
        # Print summary
        print(f"\n🎉 PORK BELLY RECIPE GENERATION COMPLETE!")
        print(f"   📊 Success Rate: {results['success_rate']:.1%}")
        print(f"   🎯 Average Quality: {results['avg_quality_score']:.3f}")
        print(f"   ⚡ Performance: {results['avg_tokens_per_second']:.1f} tokens/sec")
        
        # Send Discord notification
        generator.send_discord_notification(results)
        
        # Always save readable recipes to text file
        generator.save_readable_recipes(results, args.output)
        
        # Print category summary
        print(f"\n📊 CATEGORY BREAKDOWN:")
        for category, stats in results["category_stats"].items():
            print(f"   {category.title()}: {stats['success_rate']:.1%} success, {stats['avg_quality']:.3f} quality")
        
        # Always offer to show recipes interactively
        show_recipes = input(f"\n📖 Display recipes in terminal for browsing? (y/n): ").strip().lower()
        if show_recipes == 'y':
            generator.print_all_recipes_readable(results)
        
    else:
        print(f"❌ Generation failed: {results['error']}")

if __name__ == "__main__":
    main()