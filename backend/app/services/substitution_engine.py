from typing import List, Dict, Any, Optional
import logging
from datetime import datetime
from app.models.recipe import Ingredient
import asyncio
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger(__name__)

class SubstitutionEngine:
    def __init__(self):
        self.substitution_db = self._load_substitution_database()
        self.flavor_profiles = self._load_flavor_profiles()
        self.dietary_mappings = self._load_dietary_mappings()
        self.seasonal_ingredients = self._load_seasonal_data()
        
        # AI-powered substitution capabilities
        self.ai_model = None
        self.ai_tokenizer = None
        self._initialize_ai_model()
    
    def _initialize_ai_model(self):
        """Initialize AI model for intelligent substitutions."""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_name = "microsoft/Phi-3.5-mini-instruct"
            
            self.ai_tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )
            
            self.ai_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True
            )
            
            if self.ai_tokenizer.pad_token is None:
                self.ai_tokenizer.pad_token = self.ai_tokenizer.eos_token
                
            self.ai_pipeline = pipeline(
                "text-generation",
                model=self.ai_model,
                tokenizer=self.ai_tokenizer,
                device=0 if device == "cuda" else -1
            )
                
            logger.info("AI substitution model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load AI model: {e}")
            self.ai_model = None
    
    def _load_substitution_database(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load ingredient substitution database."""
        return {
            "butter": [
                {"substitute": "coconut oil", "ratio": "1:1", "notes": "Use refined for neutral flavor", "score": 9},
                {"substitute": "vegan butter", "ratio": "1:1", "notes": "Direct replacement", "score": 10},
                {"substitute": "applesauce", "ratio": "1:0.5", "notes": "Reduces fat, adds moisture", "score": 7},
                {"substitute": "olive oil", "ratio": "1:0.75", "notes": "For savory dishes", "score": 6},
            ],
            "eggs": [
                {"substitute": "flax eggs", "ratio": "1:1", "notes": "1 tbsp ground flax + 3 tbsp water", "score": 8},
                {"substitute": "chia eggs", "ratio": "1:1", "notes": "1 tbsp chia seeds + 3 tbsp water", "score": 8},
                {"substitute": "applesauce", "ratio": "1:0.25 cup", "notes": "For moisture in baking", "score": 7},
                {"substitute": "aquafaba", "ratio": "1:3 tbsp", "notes": "Great for binding", "score": 9},
            ],
            "milk": [
                {"substitute": "almond milk", "ratio": "1:1", "notes": "Light, nutty flavor", "score": 9},
                {"substitute": "oat milk", "ratio": "1:1", "notes": "Creamy texture", "score": 9},
                {"substitute": "coconut milk", "ratio": "1:1", "notes": "Rich, tropical flavor", "score": 8},
                {"substitute": "soy milk", "ratio": "1:1", "notes": "High protein content", "score": 8},
            ],
            "sugar": [
                {"substitute": "honey", "ratio": "1:0.75", "notes": "Reduce liquid by 1/4", "score": 8},
                {"substitute": "maple syrup", "ratio": "1:0.75", "notes": "Reduce liquid by 3 tbsp", "score": 8},
                {"substitute": "stevia", "ratio": "1:0.25 tsp", "notes": "Very sweet, adjust carefully", "score": 7},
                {"substitute": "coconut sugar", "ratio": "1:1", "notes": "Direct replacement", "score": 9},
            ],
            "flour": [
                {"substitute": "almond flour", "ratio": "1:1", "notes": "Higher fat content", "score": 8},
                {"substitute": "coconut flour", "ratio": "1:0.25", "notes": "Very absorbent, add liquid", "score": 6},
                {"substitute": "rice flour", "ratio": "1:1", "notes": "Gluten-free option", "score": 7},
                {"substitute": "oat flour", "ratio": "1:1", "notes": "Slightly denser texture", "score": 8},
            ],
        }
    
    def _load_flavor_profiles(self) -> Dict[str, Dict[str, Any]]:
        """Load flavor compatibility data."""
        return {
            "tomato": {"sweet": 3, "sour": 6, "salty": 2, "bitter": 1, "umami": 7},
            "lemon": {"sweet": 2, "sour": 9, "salty": 1, "bitter": 3, "umami": 1},
            "garlic": {"sweet": 1, "sour": 2, "salty": 3, "bitter": 2, "umami": 8},
            "onion": {"sweet": 4, "sour": 2, "salty": 2, "bitter": 1, "umami": 6},
            "basil": {"sweet": 3, "sour": 1, "salty": 2, "bitter": 4, "umami": 3},
            "oregano": {"sweet": 2, "sour": 2, "salty": 3, "bitter": 5, "umami": 4},
            "thyme": {"sweet": 2, "sour": 1, "salty": 3, "bitter": 4, "umami": 3},
            "rosemary": {"sweet": 1, "sour": 1, "salty": 3, "bitter": 6, "umami": 3},
        }
    
    def _load_dietary_mappings(self) -> Dict[str, Dict[str, List[str]]]:
        """Load dietary restriction mappings."""
        return {
            "vegan": {
                "avoid": ["butter", "eggs", "milk", "cheese", "meat", "fish", "honey"],
                "replace_with": ["vegan butter", "flax eggs", "plant milk", "vegan cheese", "tofu", "tempeh", "maple syrup"]
            },
            "gluten-free": {
                "avoid": ["flour", "wheat", "barley", "rye", "bread", "pasta"],
                "replace_with": ["rice flour", "almond flour", "gluten-free bread", "rice pasta"]
            },
            "keto": {
                "avoid": ["sugar", "flour", "rice", "pasta", "potato", "bread"],
                "replace_with": ["stevia", "almond flour", "cauliflower rice", "zucchini noodles", "cauliflower", "cloud bread"]
            },
            "paleo": {
                "avoid": ["grains", "legumes", "dairy", "sugar", "processed foods"],
                "replace_with": ["vegetables", "fruits", "nuts", "seeds", "coconut", "honey"]
            },
            "low-sodium": {
                "avoid": ["salt", "soy sauce", "canned foods", "processed meats"],
                "replace_with": ["herbs", "spices", "lemon juice", "vinegar", "fresh ingredients"]
            }
        }
    
    def _load_seasonal_data(self) -> Dict[str, List[str]]:
        """Load seasonal ingredient availability."""
        return {
            "spring": ["asparagus", "peas", "lettuce", "spinach", "radishes", "strawberries"],
            "summer": ["tomatoes", "corn", "zucchini", "berries", "peaches", "herbs"],
            "fall": ["pumpkin", "squash", "apples", "pears", "root vegetables", "cranberries"],
            "winter": ["citrus", "cabbage", "potatoes", "onions", "stored apples", "pomegranates"]
        }
    
    async def find_substitutions(self, request) -> Dict[str, Any]:
        """Find substitutions for a given ingredient using both database and AI."""
        ingredient = request.ingredient.lower()
        
        # Direct database lookup
        substitutes = []
        if ingredient in self.substitution_db:
            substitutes = self.substitution_db[ingredient].copy()
        else:
            # Fuzzy matching for similar ingredients
            substitutes = self._find_similar_substitutions(ingredient)
        
        # AI-powered enhancement for better substitutions
        if self.ai_model and (not substitutes or len(substitutes) < 3):
            ai_substitutes = await self._get_ai_substitutions(request)
            substitutes.extend(ai_substitutes)
        
        # Filter based on dietary restrictions
        if request.dietary_restrictions:
            substitutes = self._filter_by_dietary_restrictions(
                substitutes, request.dietary_restrictions
            )
        
        # Filter based on available ingredients
        if request.available_ingredients:
            substitutes = self._filter_by_availability(
                substitutes, request.available_ingredients
            )
        
        # Sort by compatibility score
        substitutes.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Generate enhanced notes with AI insights
        notes = await self._generate_enhanced_substitution_notes(ingredient, request, substitutes)
        
        return {
            "original_ingredient": request.ingredient,
            "substitutes": substitutes[:5],  # Top 5 substitutes
            "notes": notes,
            "confidence": "high" if len(substitutes) >= 3 else "medium"
        }
    
    async def _get_ai_substitutions(self, request) -> List[Dict[str, Any]]:
        """Get AI-powered substitution suggestions."""
        if not self.ai_model:
            return []
        
        try:
            # Create intelligent prompt for substitution
            system_prompt = "You are an expert culinary scientist. Provide ingredient substitutions with precise measurements and ratios."
            
            context_parts = []
            if request.dietary_restrictions:
                context_parts.append(f"Dietary restrictions: {', '.join(request.dietary_restrictions)}")
            if hasattr(request, 'recipe_context') and request.recipe_context:
                context_parts.append(f"Recipe context: {request.recipe_context}")
            if hasattr(request, 'available_ingredients') and request.available_ingredients:
                context_parts.append(f"Available ingredients: {', '.join(request.available_ingredients)}")
            
            context = " | ".join(context_parts) if context_parts else "General cooking"
            
            user_prompt = f"""Find the best substitutes for "{request.ingredient}" in cooking.
Context: {context}

Provide exactly 3 substitutes in this format:
1. [substitute name] - [ratio like 1:1] - [brief note about usage]
2. [substitute name] - [ratio like 1:1] - [brief note about usage]  
3. [substitute name] - [ratio like 1:1] - [brief note about usage]

Only respond with the numbered list, no other text."""

            prompt = f"<|system|>\n{system_prompt}<|end|>\n<|user|>\n{user_prompt}<|end|>\n<|assistant|>\n"
            
            result = self.ai_pipeline(
                prompt,
                max_new_tokens=300,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )
            
            ai_response = result[0]['generated_text'].strip()
            return self._parse_ai_substitutions(ai_response)
            
        except Exception as e:
            logger.error(f"AI substitution generation failed: {e}")
            return []
    
    def _parse_ai_substitutions(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured substitution format."""
        substitutes = []
        lines = ai_response.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or not any(line.startswith(str(i)) for i in range(1, 10)):
                continue
                
            try:
                # Parse format: "1. ingredient - ratio - note"
                parts = line.split(' - ', 2)
                if len(parts) >= 2:
                    # Extract substitute name (remove number prefix)
                    substitute_part = parts[0].split('.', 1)[1].strip() if '.' in parts[0] else parts[0].strip()
                    ratio = parts[1].strip()
                    note = parts[2].strip() if len(parts) > 2 else "AI suggested substitute"
                    
                    substitutes.append({
                        "substitute": substitute_part,
                        "ratio": ratio,
                        "notes": note,
                        "score": 7,  # AI suggestions get good score
                        "source": "ai"
                    })
            except Exception as e:
                logger.warning(f"Failed to parse AI substitution line: {line}, error: {e}")
                continue
        
        return substitutes[:3]  # Return max 3 AI suggestions
    
    async def _generate_enhanced_substitution_notes(self, ingredient: str, request, substitutes: List[Dict]) -> List[str]:
        """Generate enhanced notes using AI insights."""
        base_notes = self._generate_substitution_notes(ingredient, request)
        
        # Add AI-generated contextual notes if available
        if self.ai_model and len(substitutes) > 0:
            try:
                ai_notes = await self._get_ai_substitution_tips(ingredient, request, substitutes[:3])
                base_notes.extend(ai_notes)
            except Exception as e:
                logger.error(f"Failed to generate AI notes: {e}")
        
        return base_notes
    
    async def _get_ai_substitution_tips(self, ingredient: str, request, top_substitutes: List[Dict]) -> List[str]:
        """Get AI-generated tips for substitutions."""
        if not self.ai_model:
            return []
        
        try:
            substitutes_text = ", ".join([s["substitute"] for s in top_substitutes])
            
            prompt = f"""<|system|>
You are a culinary expert. Provide 2-3 concise cooking tips for ingredient substitutions.<|end|>
<|user|>
When substituting {ingredient} with {substitutes_text}, what are the most important tips to ensure success? 
Provide exactly 2-3 brief tips, each starting with "•".<|end|>
<|assistant|>
"""
            
            result = self.ai_pipeline(
                prompt,
                max_new_tokens=200,
                temperature=0.4,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )
            
            ai_response = result[0]['generated_text'].strip()
            
            # Extract bullet points
            tips = []
            for line in ai_response.split('\n'):
                line = line.strip()
                if line.startswith('•') or line.startswith('-'):
                    tips.append(line[1:].strip())
            
            return tips[:3]
            
        except Exception as e:
            logger.error(f"AI tip generation failed: {e}")
            return []
    
    def _find_similar_substitutions(self, ingredient: str) -> List[Dict[str, Any]]:
        """Find substitutions for ingredients not in the direct database."""
        # Simple keyword matching - in production, use more sophisticated NLP
        generic_substitutes = []
        
        if any(word in ingredient for word in ["meat", "chicken", "beef", "pork"]):
            generic_substitutes = [
                {"substitute": "tofu", "ratio": "1:1", "notes": "Firm tofu, well-pressed", "score": 8},
                {"substitute": "tempeh", "ratio": "1:1", "notes": "Nutty flavor, firm texture", "score": 7},
                {"substitute": "mushrooms", "ratio": "1:1", "notes": "Portobello or shiitake", "score": 6},
            ]
        elif "cheese" in ingredient:
            generic_substitutes = [
                {"substitute": "nutritional yeast", "ratio": "1:0.5", "notes": "Cheesy flavor", "score": 7},
                {"substitute": "cashew cream", "ratio": "1:1", "notes": "Creamy texture", "score": 8},
                {"substitute": "vegan cheese", "ratio": "1:1", "notes": "Direct replacement", "score": 9},
            ]
        
        return generic_substitutes
    
    def _filter_by_dietary_restrictions(self, substitutes: List[Dict], restrictions: List[str]) -> List[Dict]:
        """Filter substitutes based on dietary restrictions."""
        filtered = []
        
        for substitute in substitutes:
            substitute_name = substitute["substitute"].lower()
            is_suitable = True
            
            for restriction in restrictions:
                restriction_lower = restriction.lower()
                if restriction_lower in self.dietary_mappings:
                    avoid_list = self.dietary_mappings[restriction_lower]["avoid"]
                    if any(avoid_item in substitute_name for avoid_item in avoid_list):
                        is_suitable = False
                        break
            
            if is_suitable:
                filtered.append(substitute)
        
        return filtered
    
    def _filter_by_availability(self, substitutes: List[Dict], available: List[str]) -> List[Dict]:
        """Filter substitutes based on available ingredients."""
        available_lower = [item.lower() for item in available]
        
        return [
            sub for sub in substitutes
            if any(avail in sub["substitute"].lower() for avail in available_lower)
        ]
    
    def _generate_substitution_notes(self, ingredient: str, request) -> List[str]:
        """Generate helpful notes for substitutions."""
        notes = []
        
        if request.recipe_context:
            if "baking" in request.recipe_context.lower():
                notes.append("For baking, maintain proper fat/liquid ratios")
            elif "frying" in request.recipe_context.lower():
                notes.append("Consider smoke point when substituting oils")
        
        if request.dietary_restrictions:
            if "vegan" in request.dietary_restrictions:
                notes.append("All substitutes are plant-based")
            if "gluten-free" in request.dietary_restrictions:
                notes.append("Check that substitute ingredients are certified gluten-free")
        
        notes.append("Taste and adjust seasonings as needed")
        notes.append("Substitutions may slightly alter texture or flavor")
        
        return notes
    
    def check_flavor_compatibility(self, ingredient1: str, ingredient2: str) -> Dict[str, Any]:
        """Check flavor compatibility between two ingredients."""
        profile1 = self.flavor_profiles.get(ingredient1.lower(), {})
        profile2 = self.flavor_profiles.get(ingredient2.lower(), {})
        
        if not profile1 or not profile2:
            return {
                "score": 5,  # Neutral score for unknown ingredients
                "explanation": "Limited flavor profile data available"
            }
        
        # Calculate compatibility based on flavor profile similarity
        compatibility_score = 0
        total_attributes = 0
        
        for attribute in ["sweet", "sour", "salty", "bitter", "umami"]:
            if attribute in profile1 and attribute in profile2:
                # Higher score for complementary levels (not necessarily identical)
                diff = abs(profile1[attribute] - profile2[attribute])
                compatibility_score += max(0, 10 - diff)
                total_attributes += 1
        
        if total_attributes > 0:
            final_score = compatibility_score / total_attributes
        else:
            final_score = 5
        
        explanation = self._generate_compatibility_explanation(final_score, profile1, profile2)
        
        return {
            "score": round(final_score, 1),
            "explanation": explanation
        }
    
    def _generate_compatibility_explanation(self, score: float, profile1: Dict, profile2: Dict) -> str:
        """Generate explanation for compatibility score."""
        if score >= 8:
            return "Excellent flavor pairing with complementary taste profiles"
        elif score >= 6:
            return "Good compatibility, flavors work well together"
        elif score >= 4:
            return "Moderate compatibility, may need careful balancing"
        else:
            return "Challenging pairing, consider adjusting proportions"
    
    def get_seasonal_alternatives(self, ingredient: str, season: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get seasonal alternatives for an ingredient."""
        if not season:
            # Determine current season based on month
            month = datetime.now().month
            if month in [3, 4, 5]:
                season = "spring"
            elif month in [6, 7, 8]:
                season = "summer"
            elif month in [9, 10, 11]:
                season = "fall"
            else:
                season = "winter"
        
        seasonal_items = self.seasonal_ingredients.get(season.lower(), [])
        
        # Find alternatives that are in season
        alternatives = []
        for item in seasonal_items:
            if item != ingredient.lower():
                alternatives.append({
                    "ingredient": item,
                    "season": season,
                    "benefits": f"In season during {season}, likely fresher and more affordable"
                })
        
        return alternatives[:5]  # Return top 5 alternatives