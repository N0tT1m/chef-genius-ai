#!/usr/bin/env python3
"""
🎯 DUAL-MODE TOKENIZER
Support both B2B enterprise tokens AND simple format
Automatically detects which format to use based on training data
"""

import json
import re
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

@dataclass
class TokenizerResult:
    """Result from tokenization process."""
    formatted_prompt: str
    token_count: int
    within_limit: bool
    mode_used: str  # "b2b_enterprise" or "simple"
    business_context: Optional[str] = None

class DualModeTokenizer:
    """Tokenizer that supports both B2B enterprise and simple formats."""
    
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        
        # B2B enterprise token vocabulary
        self.enterprise_tokens = {
            # Business structure tokens
            "[BUSINESS_REQUEST]": 1000,
            "[/BUSINESS_REQUEST]": 1001,
            "[BUSINESS_TYPE]": 1002,
            "[/BUSINESS_TYPE]": 1003,
            "[SERVICE_STYLE]": 1004,
            "[/SERVICE_STYLE]": 1005,
            "[VOLUME]": 1006,
            "[/VOLUME]": 1007,
            "[COST_TARGET]": 1008,
            "[/COST_TARGET]": 1009,
            "[SKILL_LEVEL]": 1010,
            "[/SKILL_LEVEL]": 1011,
            "[MEAL_STRUCTURE]": 1012,
            "[/MEAL_STRUCTURE]": 1013,
            
            # Recipe structure tokens
            "[RECIPE_START]": 1014,
            "[RECIPE_END]": 1015,
            "[TITLE_START]": 1016,
            "[TITLE_END]": 1017,
            "[INGREDIENTS_START]": 1018,
            "[INGREDIENTS_END]": 1019,
            "[INGREDIENT]": 1020,
            "[/INGREDIENT]": 1021,
            "[INSTRUCTIONS_START]": 1022,
            "[INSTRUCTIONS_END]": 1023,
            "[STEP]": 1024,
            "[/STEP]": 1025,
            "[TECHNIQUE]": 1026,
            "[/TECHNIQUE]": 1027,
            
            # Business-specific tokens
            "[EQUIPMENT_START]": 1028,
            "[EQUIPMENT_END]": 1029,
            "[BUSINESS_NOTES_START]": 1030,
            "[BUSINESS_NOTES_END]": 1031,
            "[BUSINESS_INFO_START]": 1032,
            "[BUSINESS_INFO_END]": 1033,
            "[REQUIREMENTS]": 1034,
            "[/REQUIREMENTS]": 1035,
            "[PREP_TIME]": 1036,
            "[/PREP_TIME]": 1037,
        }
        
        # Simple vocabulary for regular recipes
        self.simple_vocab = self._create_simple_vocab()
        
        # Business context detection
        self.b2b_keywords = [
            "restaurant", "catering", "commercial", "business", "food service",
            "covers", "servings", "volume", "cost", "budget", "staff", "kitchen",
            "meal kit", "institutional", "school", "hospital", "corporate",
            "fine dining", "fast casual", "line cook", "professional chef"
        ]
        
        print(f"🎯 Dual-mode tokenizer initialized:")
        print(f"   📊 Enterprise tokens: {len(self.enterprise_tokens)}")
        print(f"   📝 Simple vocab: {len(self.simple_vocab)}")
        print(f"   🎯 Max length: {self.max_length}")
    
    def _create_simple_vocab(self) -> Dict[str, int]:
        """Create simple vocabulary for regular recipes."""
        vocab = {}
        
        # Special tokens
        special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
        for i, token in enumerate(special_tokens):
            vocab[token] = i
        
        # Common recipe words
        recipe_words = [
            # Basic words
            "the", "and", "a", "to", "in", "of", "for", "with", "on", "at", "is", "or",
            "add", "mix", "cook", "heat", "bake", "fry", "boil", "serve", "until", "about",
            
            # Measurements
            "cup", "cups", "tablespoon", "tablespoons", "teaspoon", "teaspoons",
            "tbsp", "tsp", "ounce", "ounces", "oz", "pound", "pounds", "lb",
            "gram", "grams", "g", "liter", "ml", "inch", "inches",
            
            # Ingredients
            "chicken", "beef", "pork", "fish", "salmon", "eggs", "milk", "butter",
            "oil", "salt", "pepper", "garlic", "onion", "cheese", "flour", "sugar",
            "water", "rice", "pasta", "bread", "potato", "tomato",
            
            # Cooking methods
            "roast", "grill", "steam", "saute", "brown", "season", "chop", "slice",
            "dice", "mince", "blend", "whisk", "stir", "combine", "prepare",
            
            # Recipe structure
            "ingredients", "instructions", "recipe", "create", "make"
        ]
        
        for i, word in enumerate(recipe_words):
            vocab[word] = len(special_tokens) + i
        
        return vocab
    
    def detect_mode(self, text: str, force_mode: Optional[str] = None) -> str:
        """Detect whether to use B2B or simple mode."""
        if force_mode:
            return force_mode
        
        text_lower = text.lower()
        
        # Check for B2B indicators
        b2b_score = sum(1 for keyword in self.b2b_keywords if keyword in text_lower)
        
        # Check for enterprise tokens already present
        enterprise_token_present = any(token in text for token in self.enterprise_tokens.keys())
        
        # Decision logic
        if enterprise_token_present or b2b_score >= 2:
            return "b2b_enterprise"
        elif b2b_score == 1:
            # Single B2B keyword - could go either way, prefer simple for compatibility
            return "simple"
        else:
            return "simple"
    
    def format_prompt(
        self, 
        request: str, 
        mode: Optional[str] = None,
        business_context: Optional[Dict] = None
    ) -> TokenizerResult:
        """Format prompt in the appropriate mode."""
        
        detected_mode = self.detect_mode(request, mode)
        
        if detected_mode == "b2b_enterprise":
            return self._format_b2b_prompt(request, business_context)
        else:
            return self._format_simple_prompt(request)
    
    def _format_simple_prompt(self, request: str) -> TokenizerResult:
        """Format prompt in simple mode (matches your training data)."""
        
        # Clean up request
        if not request.lower().startswith("create"):
            request = f"create {request.lower()}"
        
        # Simple format matching training data
        formatted = f"{request.capitalize()}\n\nIngredients:"
        
        # Tokenize with simple vocab
        tokens = self._tokenize_simple(formatted)
        token_count = len(tokens)
        
        return TokenizerResult(
            formatted_prompt=formatted,
            token_count=token_count,
            within_limit=token_count <= self.max_length,
            mode_used="simple"
        )
    
    def _format_b2b_prompt(
        self, 
        request: str, 
        business_context: Optional[Dict] = None
    ) -> TokenizerResult:
        """Format prompt in B2B enterprise mode with special tokens."""
        
        # Extract or default business context
        if business_context is None:
            business_context = self._extract_business_context(request)
        
        # Create B2B prompt with enterprise tokens
        formatted = f"""[BUSINESS_REQUEST]
[BUSINESS_TYPE]{business_context.get('business_type', 'Commercial Kitchen')}[/BUSINESS_TYPE]
[SERVICE_STYLE]{business_context.get('service_style', 'Professional Service')}[/SERVICE_STYLE]
[VOLUME]{business_context.get('volume', '100 servings')}[/VOLUME]
[COST_TARGET]{business_context.get('cost_target', 'Mid-range')}[/COST_TARGET]
[SKILL_LEVEL]{business_context.get('skill_level', 'Professional')}[/SKILL_LEVEL]
[MEAL_STRUCTURE]{business_context.get('meal_structure', 'Complete Meal')}[/MEAL_STRUCTURE]

{request}

[REQUIREMENTS]
- Food cost control and portion consistency
- Equipment efficiency and workflow optimization
- Food safety and temperature control compliance
- Scalable preparation methods for volume production
[/REQUIREMENTS]
[/BUSINESS_REQUEST]

Generate enterprise recipe:"""
        
        # Tokenize with enterprise tokens
        tokens = self._tokenize_enterprise(formatted)
        token_count = len(tokens)
        
        # Check if we need to truncate
        if token_count > self.max_length:
            formatted = self._truncate_b2b_prompt(formatted, business_context)
            tokens = self._tokenize_enterprise(formatted)
            token_count = len(tokens)
        
        return TokenizerResult(
            formatted_prompt=formatted,
            token_count=token_count,
            within_limit=token_count <= self.max_length,
            mode_used="b2b_enterprise",
            business_context=business_context.get('business_type', 'Commercial')
        )
    
    def _extract_business_context(self, request: str) -> Dict[str, str]:
        """Extract business context from request text."""
        request_lower = request.lower()
        
        # Business type detection
        if "restaurant" in request_lower:
            if "fine dining" in request_lower:
                business_type = "Fine Dining Restaurant"
                service_style = "Fine Dining"
                cost_target = "Premium"
                skill_level = "Professional Chef"
            elif "fast casual" in request_lower or "quick" in request_lower:
                business_type = "Fast Casual Restaurant"
                service_style = "Fast Casual"
                cost_target = "Budget"
                skill_level = "Line Cook"
            else:
                business_type = "Restaurant"
                service_style = "Casual Dining"
                cost_target = "Mid-range"
                skill_level = "Professional"
        elif "catering" in request_lower:
            business_type = "Catering Service"
            service_style = "Catering"
            cost_target = "Mid-range"
            skill_level = "Professional"
        elif "meal kit" in request_lower:
            business_type = "Meal Kit Service"
            service_style = "Home Cooking"
            cost_target = "Mid-range"
            skill_level = "Home Cook"
        elif "school" in request_lower or "institutional" in request_lower:
            business_type = "Institutional Kitchen"
            service_style = "Institutional"
            cost_target = "Budget"
            skill_level = "Institutional Cook"
        else:
            business_type = "Commercial Kitchen"
            service_style = "Professional Service"
            cost_target = "Mid-range"
            skill_level = "Professional"
        
        # Volume detection
        volume_match = re.search(r'(\d+)\s*(serving|cover|guest|people)', request_lower)
        if volume_match:
            volume = f"{volume_match.group(1)} servings"
        else:
            volume = "100 servings"
        
        # Meal structure detection
        if "protein" in request_lower and "side" in request_lower:
            meal_structure = "1 protein + 2 sides"
        else:
            meal_structure = "Complete Meal"
        
        return {
            "business_type": business_type,
            "service_style": service_style,
            "volume": volume,
            "cost_target": cost_target,
            "skill_level": skill_level,
            "meal_structure": meal_structure
        }
    
    def _truncate_b2b_prompt(self, prompt: str, business_context: Dict) -> str:
        """Truncate B2B prompt to fit within token limit."""
        
        # Simplified B2B prompt
        simplified = f"""[BUSINESS_REQUEST]
[BUSINESS_TYPE]{business_context.get('business_type', 'Commercial')}[/BUSINESS_TYPE]
[VOLUME]{business_context.get('volume', '100 servings')}[/VOLUME]
[COST_TARGET]{business_context.get('cost_target', 'Mid-range')}[/COST_TARGET]

Create commercial recipe optimized for business use.
[/BUSINESS_REQUEST]

Generate recipe:"""
        
        return simplified
    
    def _tokenize_simple(self, text: str) -> List[int]:
        """Tokenize text using simple vocabulary."""
        # Basic tokenization
        words = re.findall(r'\w+', text.lower())
        tokens = []
        
        for word in words:
            token_id = self.simple_vocab.get(word, self.simple_vocab["<UNK>"])
            tokens.append(token_id)
        
        return tokens[:self.max_length]
    
    def _tokenize_enterprise(self, text: str) -> List[int]:
        """Tokenize text using enterprise tokens + simple vocab."""
        tokens = []
        
        # First, extract enterprise tokens
        remaining_text = text
        
        for token, token_id in self.enterprise_tokens.items():
            while token in remaining_text:
                # Find position of token
                pos = remaining_text.find(token)
                
                # Tokenize text before the enterprise token
                before_text = remaining_text[:pos]
                if before_text.strip():
                    before_tokens = self._tokenize_simple(before_text)
                    tokens.extend(before_tokens)
                
                # Add enterprise token
                tokens.append(token_id)
                
                # Continue with remaining text
                remaining_text = remaining_text[pos + len(token):]
        
        # Tokenize any remaining text
        if remaining_text.strip():
            remaining_tokens = self._tokenize_simple(remaining_text)
            tokens.extend(remaining_tokens)
        
        return tokens[:self.max_length]
    
    def batch_process(
        self, 
        requests: List[str], 
        mode: Optional[str] = None
    ) -> List[TokenizerResult]:
        """Process multiple requests efficiently."""
        
        results = []
        for request in requests:
            result = self.format_prompt(request, mode)
            results.append(result)
        
        return results
    
    def validate_training_compatibility(self, dataset_sample: List[Dict]) -> Dict:
        """Validate that tokenizer is compatible with training data."""
        
        print("🔍 TRAINING COMPATIBILITY VALIDATION")
        print("=" * 40)
        
        simple_count = 0
        b2b_count = 0
        token_stats = []
        
        for sample in dataset_sample[:100]:  # Test first 100 samples
            input_text = sample.get('input', '')
            mode = self.detect_mode(input_text)
            
            result = self.format_prompt(input_text, mode)
            token_stats.append(result.token_count)
            
            if result.mode_used == "simple":
                simple_count += 1
            else:
                b2b_count += 1
        
        avg_tokens = sum(token_stats) / len(token_stats) if token_stats else 0
        within_limit = sum(1 for t in token_stats if t <= self.max_length)
        
        validation_results = {
            "total_samples": len(token_stats),
            "simple_mode_used": simple_count,
            "b2b_mode_used": b2b_count,
            "avg_token_count": avg_tokens,
            "within_limit_count": within_limit,
            "within_limit_percentage": within_limit / len(token_stats) * 100 if token_stats else 0,
            "max_tokens": max(token_stats) if token_stats else 0,
            "min_tokens": min(token_stats) if token_stats else 0
        }
        
        print(f"📊 Samples tested: {validation_results['total_samples']}")
        print(f"📝 Simple mode: {validation_results['simple_mode_used']}")
        print(f"🏢 B2B mode: {validation_results['b2b_mode_used']}")
        print(f"📏 Avg tokens: {validation_results['avg_token_count']:.1f}")
        print(f"✅ Within limit: {validation_results['within_limit_percentage']:.1f}%")
        
        return validation_results

def main():
    """Demo the dual-mode tokenizer."""
    print("🎯 DUAL-MODE TOKENIZER DEMO")
    print("=" * 40)
    
    tokenizer = DualModeTokenizer()
    
    # Test prompts - mix of simple and B2B
    test_prompts = [
        # Simple prompts (should use simple mode)
        "Create a chocolate chip cookie recipe",
        "Make a pasta with tomato sauce",
        "Create a chicken salad",
        
        # B2B prompts (should use enterprise mode)
        "Create a chicken dish with rice and vegetables for restaurant service",
        "Create a salmon recipe with two sides for fine dining 50 covers",
        "Create a catering meal for 300 corporate guests",
        "Create a meal kit with protein and two sides for family cooking",
        
        # Ambiguous prompts (will auto-detect)
        "Create a quick lunch recipe",
        "Create a budget-friendly dinner"
    ]
    
    print(f"\n🧪 Testing {len(test_prompts)} prompts...")
    
    results = tokenizer.batch_process(test_prompts)
    
    for i, (prompt, result) in enumerate(zip(test_prompts, results), 1):
        print(f"\n📝 Test {i}:")
        print(f"   Input: {prompt}")
        print(f"   Mode: {result.mode_used}")
        print(f"   Tokens: {result.token_count} {'✅' if result.within_limit else '❌'}")
        if result.business_context:
            print(f"   Context: {result.business_context}")
        print(f"   Preview: {result.formatted_prompt[:100]}...")
    
    # Summary statistics
    simple_results = [r for r in results if r.mode_used == "simple"]
    b2b_results = [r for r in results if r.mode_used == "b2b_enterprise"]
    
    print(f"\n📊 SUMMARY:")
    print(f"Simple mode: {len(simple_results)} prompts")
    print(f"B2B mode: {len(b2b_results)} prompts")
    print(f"Avg tokens (simple): {sum(r.token_count for r in simple_results) / len(simple_results):.1f}" if simple_results else "N/A")
    print(f"Avg tokens (B2B): {sum(r.token_count for r in b2b_results) / len(b2b_results):.1f}" if b2b_results else "N/A")
    print(f"All within limit: {'✅' if all(r.within_limit for r in results) else '❌'}")

if __name__ == "__main__":
    main()