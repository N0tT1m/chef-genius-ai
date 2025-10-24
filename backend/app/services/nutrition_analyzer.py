from typing import List, Dict, Any, Optional
import logging
from app.models.recipe import NutritionInfo

logger = logging.getLogger(__name__)

class NutritionAnalyzer:
    def __init__(self):
        # In production, this would connect to a nutrition database (USDA, etc.)
        self.nutrition_db = self._load_nutrition_database()
    
    def _load_nutrition_database(self) -> Dict[str, Dict[str, float]]:
        """Load nutrition data for common ingredients."""
        # Simplified nutrition database (per 100g)
        return {
            "chicken": {"calories": 165, "protein": 31, "fat": 3.6, "carbs": 0, "fiber": 0},
            "beef": {"calories": 250, "protein": 26, "fat": 15, "carbs": 0, "fiber": 0},
            "salmon": {"calories": 208, "protein": 20, "fat": 12, "carbs": 0, "fiber": 0},
            "rice": {"calories": 130, "protein": 2.7, "fat": 0.3, "carbs": 28, "fiber": 0.4},
            "pasta": {"calories": 131, "protein": 5, "fat": 1.1, "carbs": 25, "fiber": 1.8},
            "bread": {"calories": 265, "protein": 9, "fat": 3.2, "carbs": 49, "fiber": 2.7},
            "egg": {"calories": 155, "protein": 13, "fat": 11, "carbs": 1.1, "fiber": 0},
            "milk": {"calories": 42, "protein": 3.4, "fat": 1, "carbs": 5, "fiber": 0},
            "cheese": {"calories": 113, "protein": 7, "fat": 9, "carbs": 1, "fiber": 0},
            "butter": {"calories": 717, "protein": 0.9, "fat": 81, "carbs": 0.1, "fiber": 0},
            "olive oil": {"calories": 884, "protein": 0, "fat": 100, "carbs": 0, "fiber": 0},
            "onion": {"calories": 40, "protein": 1.1, "fat": 0.1, "carbs": 9.3, "fiber": 1.7},
            "garlic": {"calories": 149, "protein": 6.4, "fat": 0.5, "carbs": 33, "fiber": 2.1},
            "tomato": {"calories": 18, "protein": 0.9, "fat": 0.2, "carbs": 3.9, "fiber": 1.2},
            "carrot": {"calories": 41, "protein": 0.9, "fat": 0.2, "carbs": 9.6, "fiber": 2.8},
            "potato": {"calories": 77, "protein": 2, "fat": 0.1, "carbs": 17, "fiber": 2.2},
            "broccoli": {"calories": 34, "protein": 2.8, "fat": 0.4, "carbs": 7, "fiber": 2.6},
            "spinach": {"calories": 23, "protein": 2.9, "fat": 0.4, "carbs": 3.6, "fiber": 2.2},
            "apple": {"calories": 52, "protein": 0.3, "fat": 0.2, "carbs": 14, "fiber": 2.4},
            "banana": {"calories": 89, "protein": 1.1, "fat": 0.3, "carbs": 23, "fiber": 2.6},
            "flour": {"calories": 364, "protein": 10, "fat": 1, "carbs": 76, "fiber": 2.7},
            "sugar": {"calories": 387, "protein": 0, "fat": 0, "carbs": 100, "fiber": 0},
            "salt": {"calories": 0, "protein": 0, "fat": 0, "carbs": 0, "fiber": 0},
            "pepper": {"calories": 251, "protein": 10, "fat": 3.3, "carbs": 64, "fiber": 25},
        }
    
    def analyze_recipe(self, ingredients: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze the nutritional content of a recipe."""
        try:
            total_nutrition = {
                "calories": 0,
                "protein": 0,
                "fat": 0,
                "carbohydrates": 0,
                "fiber": 0,
                "sugar": 0,
                "sodium": 0
            }
            
            for ingredient in ingredients:
                nutrition = self._get_ingredient_nutrition(ingredient)
                for nutrient, value in nutrition.items():
                    if nutrient in total_nutrition:
                        total_nutrition[nutrient] += value
            
            # Round to 1 decimal place
            return {k: round(v, 1) for k, v in total_nutrition.items()}
            
        except Exception as e:
            logger.error(f"Nutrition analysis failed: {e}")
            return self._get_default_nutrition()
    
    def _get_ingredient_nutrition(self, ingredient: Dict[str, Any]) -> Dict[str, float]:
        """Get nutrition for a single ingredient."""
        ingredient_name = ingredient.get("name", "").lower()
        amount = ingredient.get("amount", 1)
        unit = ingredient.get("unit", "cup")
        
        # Find matching ingredient in database
        base_nutrition = None
        for key in self.nutrition_db:
            if key in ingredient_name or ingredient_name in key:
                base_nutrition = self.nutrition_db[key].copy()
                break
        
        if not base_nutrition:
            # Default values for unknown ingredients
            base_nutrition = {"calories": 50, "protein": 2, "fat": 1, "carbs": 10, "fiber": 1}
        
        # Convert amount to grams (simplified conversion)
        grams = self._convert_to_grams(amount, unit, ingredient_name)
        
        # Scale nutrition based on actual amount
        scale_factor = grams / 100  # Base nutrition is per 100g
        
        return {
            "calories": base_nutrition["calories"] * scale_factor,
            "protein": base_nutrition["protein"] * scale_factor,
            "fat": base_nutrition["fat"] * scale_factor,
            "carbohydrates": base_nutrition.get("carbs", 0) * scale_factor,
            "fiber": base_nutrition["fiber"] * scale_factor,
            "sugar": base_nutrition.get("sugar", 0) * scale_factor,
            "sodium": base_nutrition.get("sodium", 0) * scale_factor
        }
    
    def _convert_to_grams(self, amount: float, unit: str, ingredient_name: str) -> float:
        """Convert ingredient amount to grams (simplified conversion)."""
        # Simplified conversion table
        conversions = {
            "cup": 240,  # ml, varies by ingredient
            "tbsp": 15,
            "tsp": 5,
            "oz": 28.35,
            "lb": 453.6,
            "g": 1,
            "kg": 1000,
            "ml": 1,  # for liquids, assume 1ml ≈ 1g
            "l": 1000,
            "piece": 100,  # average piece weight
            "clove": 3,  # garlic clove
            "slice": 30,  # bread slice
        }
        
        # Special cases for specific ingredients
        if "garlic" in ingredient_name and "clove" in unit:
            return amount * 3
        elif "egg" in ingredient_name:
            return amount * 50  # average egg weight
        elif "onion" in ingredient_name and ("medium" in unit or "piece" in unit):
            return amount * 150
        
        # Default conversion
        multiplier = conversions.get(unit.lower(), 100)
        return amount * multiplier
    
    def _get_default_nutrition(self) -> Dict[str, float]:
        """Return default nutrition values when analysis fails."""
        return {
            "calories": 200,
            "protein": 10,
            "fat": 8,
            "carbohydrates": 25,
            "fiber": 3,
            "sugar": 5,
            "sodium": 400
        }
    
    def calculate_nutrition_per_serving(self, recipe_nutrition: Dict[str, float], servings: int) -> Dict[str, float]:
        """Calculate nutrition per serving."""
        if servings <= 0:
            servings = 1
        
        return {
            nutrient: round(value / servings, 1)
            for nutrient, value in recipe_nutrition.items()
        }
    
    def get_nutrition_score(self, nutrition: Dict[str, float]) -> Dict[str, Any]:
        """Calculate a simple nutrition score and recommendations."""
        calories = nutrition.get("calories", 0)
        protein = nutrition.get("protein", 0)
        fat = nutrition.get("fat", 0)
        carbs = nutrition.get("carbohydrates", 0)
        fiber = nutrition.get("fiber", 0)
        
        # Simple scoring (0-10 scale)
        scores = {
            "protein": min(10, (protein / calories * 100) * 2) if calories > 0 else 0,
            "fiber": min(10, fiber * 2),
            "balance": 5  # placeholder for more complex balance calculation
        }
        
        recommendations = []
        if protein < 10:
            recommendations.append("Consider adding more protein sources")
        if fiber < 3:
            recommendations.append("Add more vegetables or whole grains for fiber")
        if fat > 30:
            recommendations.append("Consider reducing fat content")
        
        return {
            "scores": scores,
            "recommendations": recommendations,
            "overall_score": round(sum(scores.values()) / len(scores), 1)
        }