from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import random
import logging
from app.services.recipe_generator import RecipeGeneratorService
from app.services.nutrition_analyzer import NutritionAnalyzer
from app.models.recipe import RecipeGenerationRequest

logger = logging.getLogger(__name__)

class MealPlannerService:
    def __init__(self):
        self.recipe_generator = RecipeGeneratorService()
        self.nutrition_analyzer = NutritionAnalyzer()
        self.meal_types = {
            "breakfast": {
                "calories_target": 300,
                "typical_cuisines": ["american", "continental", "mediterranean"],
                "common_ingredients": ["eggs", "oats", "yogurt", "fruits", "toast"]
            },
            "lunch": {
                "calories_target": 500,
                "typical_cuisines": ["mediterranean", "asian", "american", "mexican"],
                "common_ingredients": ["salad", "sandwich", "soup", "rice", "pasta"]
            },
            "dinner": {
                "calories_target": 600,
                "typical_cuisines": ["italian", "asian", "american", "indian", "mexican"],
                "common_ingredients": ["chicken", "fish", "vegetables", "rice", "pasta"]
            },
            "snack": {
                "calories_target": 200,
                "typical_cuisines": ["healthy", "mediterranean"],
                "common_ingredients": ["nuts", "fruits", "yogurt", "hummus"]
            }
        }
    
    async def generate_meal_plan(self, request) -> Dict[str, Any]:
        """Generate a comprehensive meal plan."""
        try:
            start_date = datetime.now().date()
            end_date = start_date + timedelta(days=request.days - 1)
            
            # Generate meals for each day
            meals = []
            total_nutrition = {"calories": 0, "protein": 0, "fat": 0, "carbohydrates": 0}
            
            for day in range(request.days):
                current_date = start_date + timedelta(days=day)
                daily_meals = await self._generate_daily_meals(
                    current_date, request.meals_per_day, request
                )
                meals.extend(daily_meals)
                
                # Accumulate nutrition
                for meal in daily_meals:
                    nutrition = meal.get("nutrition", {})
                    for nutrient in total_nutrition:
                        total_nutrition[nutrient] += nutrition.get(nutrient, 0)
            
            # Generate shopping list
            shopping_list = self._generate_shopping_list(meals)
            
            # Create prep schedule
            prep_schedule = self._generate_prep_schedule(meals, request.days)
            
            # Calculate nutrition summary
            nutrition_summary = self._calculate_nutrition_summary(total_nutrition, request.days)
            
            # Estimate cost
            estimated_cost = self._estimate_cost(shopping_list, request.budget_range)
            
            return {
                "plan_name": f"{request.days}-Day Meal Plan",
                "start_date": datetime.combine(start_date, datetime.min.time()),
                "end_date": datetime.combine(end_date, datetime.min.time()),
                "total_days": request.days,
                "meals": meals,
                "shopping_list": shopping_list,
                "nutrition_summary": nutrition_summary,
                "prep_schedule": prep_schedule,
                "estimated_cost": estimated_cost
            }
            
        except Exception as e:
            logger.error(f"Meal plan generation failed: {e}")
            raise
    
    async def _generate_daily_meals(self, date: datetime.date, meal_types: List[str], request) -> List[Dict[str, Any]]:
        """Generate meals for a single day."""
        daily_meals = []
        
        for meal_type in meal_types:
            meal_info = self.meal_types.get(meal_type, self.meal_types["lunch"])
            
            # Create recipe generation request for this meal
            recipe_request = RecipeGenerationRequest(
                dietary_restrictions=request.dietary_restrictions,
                cuisine=self._select_cuisine(meal_info["typical_cuisines"], request.cuisine_preferences),
                cooking_time=self._get_cooking_time_for_meal(meal_type, request.prep_time_limit),
                difficulty=request.cooking_skill,
                servings=request.people,
                meal_type=meal_type,
                ingredients=self._suggest_ingredients_for_meal(meal_type, meal_info)
            )
            
            # Generate recipe
            recipe = await self.recipe_generator.generate_recipe(recipe_request)
            
            # Analyze nutrition
            nutrition = self.nutrition_analyzer.analyze_recipe(recipe.ingredients)
            per_serving_nutrition = self.nutrition_analyzer.calculate_nutrition_per_serving(
                nutrition, recipe.servings
            )
            
            # Create meal entry
            meal = {
                "id": len(daily_meals) + 1,
                "date": date.isoformat(),
                "meal_type": meal_type,
                "recipe": {
                    "title": recipe.title,
                    "description": recipe.description,
                    "ingredients": recipe.ingredients,
                    "instructions": recipe.instructions,
                    "prep_time": recipe.prep_time,
                    "cook_time": recipe.cook_time,
                    "servings": recipe.servings,
                    "difficulty": recipe.difficulty,
                    "cuisine": recipe.cuisine
                },
                "nutrition": per_serving_nutrition,
                "prep_notes": self._generate_prep_notes(meal_type, recipe)
            }
            
            daily_meals.append(meal)
        
        return daily_meals
    
    def _select_cuisine(self, typical_cuisines: List[str], preferences: Optional[List[str]]) -> str:
        """Select cuisine for a meal."""
        if preferences:
            # Find overlap between typical and preferred cuisines
            overlap = list(set(typical_cuisines) & set(preferences))
            if overlap:
                return random.choice(overlap)
            return random.choice(preferences)
        
        return random.choice(typical_cuisines)
    
    def _get_cooking_time_for_meal(self, meal_type: str, prep_time_limit: Optional[int]) -> str:
        """Get appropriate cooking time for meal type."""
        if prep_time_limit:
            return f"under {prep_time_limit} minutes"
        
        time_limits = {
            "breakfast": "under 20 minutes",
            "lunch": "under 30 minutes", 
            "dinner": "under 60 minutes",
            "snack": "under 15 minutes"
        }
        
        return time_limits.get(meal_type, "under 30 minutes")
    
    def _suggest_ingredients_for_meal(self, meal_type: str, meal_info: Dict[str, Any]) -> List[str]:
        """Suggest appropriate ingredients for a meal type."""
        return random.sample(meal_info["common_ingredients"], min(3, len(meal_info["common_ingredients"])))
    
    def _generate_prep_notes(self, meal_type: str, recipe) -> List[str]:
        """Generate preparation notes for a meal."""
        notes = []
        
        if recipe.prep_time and recipe.prep_time > 20:
            notes.append("Consider prepping ingredients the night before")
        
        if meal_type == "breakfast" and recipe.cook_time and recipe.cook_time > 15:
            notes.append("Can be partially prepared ahead for quick morning assembly")
        
        if "marinade" in str(recipe.instructions).lower():
            notes.append("Requires advance planning for marinating")
        
        return notes
    
    def _generate_shopping_list(self, meals: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate consolidated shopping list from meals."""
        shopping_list = {
            "produce": [],
            "meat_seafood": [],
            "dairy": [],
            "pantry": [],
            "spices": [],
            "other": []
        }
        
        # Consolidate ingredients from all meals
        ingredient_totals = {}
        
        for meal in meals:
            recipe = meal.get("recipe", {})
            ingredients = recipe.get("ingredients", [])
            
            for ingredient in ingredients:
                name = ingredient.get("name", "").lower()
                amount = ingredient.get("amount", 0)
                unit = ingredient.get("unit", "")
                
                key = f"{name}_{unit}"
                if key in ingredient_totals:
                    ingredient_totals[key]["amount"] += amount
                else:
                    ingredient_totals[key] = {
                        "name": name,
                        "amount": amount,
                        "unit": unit,
                        "category": self._categorize_ingredient(name)
                    }
        
        # Organize by category
        for ingredient in ingredient_totals.values():
            category = ingredient.pop("category")
            shopping_list[category].append(ingredient)
        
        # Sort each category
        for category in shopping_list:
            shopping_list[category].sort(key=lambda x: x["name"])
        
        return shopping_list
    
    def _categorize_ingredient(self, ingredient_name: str) -> str:
        """Categorize ingredients for shopping list organization."""
        produce_keywords = ["onion", "garlic", "tomato", "lettuce", "carrot", "potato", "apple", "lemon"]
        meat_keywords = ["chicken", "beef", "pork", "fish", "salmon", "shrimp"]
        dairy_keywords = ["milk", "cheese", "butter", "yogurt", "cream"]
        spice_keywords = ["salt", "pepper", "oregano", "basil", "thyme", "cumin"]
        
        ingredient_lower = ingredient_name.lower()
        
        if any(keyword in ingredient_lower for keyword in produce_keywords):
            return "produce"
        elif any(keyword in ingredient_lower for keyword in meat_keywords):
            return "meat_seafood"
        elif any(keyword in ingredient_lower for keyword in dairy_keywords):
            return "dairy"
        elif any(keyword in ingredient_lower for keyword in spice_keywords):
            return "spices"
        elif any(keyword in ingredient_lower for keyword in ["rice", "pasta", "flour", "oil", "vinegar"]):
            return "pantry"
        else:
            return "other"
    
    def _generate_prep_schedule(self, meals: List[Dict[str, Any]], days: int) -> List[Dict[str, Any]]:
        """Generate a preparation schedule to optimize cooking efficiency."""
        prep_schedule = []
        
        # Group meals by day
        meals_by_day = {}
        for meal in meals:
            date = meal["date"]
            if date not in meals_by_day:
                meals_by_day[date] = []
            meals_by_day[date].append(meal)
        
        # Generate prep tasks
        for date, daily_meals in meals_by_day.items():
            # Day-before prep
            prep_schedule.append({
                "date": date,
                "time": "evening_before",
                "tasks": [
                    "Review tomorrow's recipes",
                    "Defrost any frozen ingredients",
                    "Prep vegetables that can be cut ahead"
                ]
            })
            
            # Morning prep
            morning_tasks = []
            for meal in daily_meals:
                if meal["meal_type"] == "breakfast":
                    morning_tasks.append(f"Prepare {meal['recipe']['title']}")
                else:
                    morning_tasks.append(f"Prep ingredients for {meal['recipe']['title']}")
            
            prep_schedule.append({
                "date": date,
                "time": "morning",
                "tasks": morning_tasks
            })
        
        return prep_schedule
    
    def _calculate_nutrition_summary(self, total_nutrition: Dict[str, float], days: int) -> Dict[str, Any]:
        """Calculate nutrition summary for the entire meal plan."""
        daily_average = {
            nutrient: round(value / days, 1)
            for nutrient, value in total_nutrition.items()
        }
        
        # Calculate percentages of daily recommended values (simplified)
        recommended_daily = {
            "calories": 2000,
            "protein": 50,
            "fat": 65,
            "carbohydrates": 300
        }
        
        percentages = {}
        for nutrient, value in daily_average.items():
            if nutrient in recommended_daily:
                percentages[f"{nutrient}_percent"] = round(
                    (value / recommended_daily[nutrient]) * 100, 1
                )
        
        return {
            "total": total_nutrition,
            "daily_average": daily_average,
            "daily_percentages": percentages,
            "assessment": self._assess_nutrition_balance(daily_average)
        }
    
    def _assess_nutrition_balance(self, daily_nutrition: Dict[str, float]) -> Dict[str, str]:
        """Assess the nutritional balance of the meal plan."""
        assessment = {"overall": "balanced"}
        
        calories = daily_nutrition.get("calories", 0)
        protein = daily_nutrition.get("protein", 0)
        fat = daily_nutrition.get("fat", 0)
        carbs = daily_nutrition.get("carbohydrates", 0)
        
        # Simple assessment rules
        if calories < 1500:
            assessment["calories"] = "low"
        elif calories > 2500:
            assessment["calories"] = "high"
        else:
            assessment["calories"] = "adequate"
        
        protein_percent = (protein * 4 / calories * 100) if calories > 0 else 0
        if protein_percent < 10:
            assessment["protein"] = "low"
        elif protein_percent > 35:
            assessment["protein"] = "high"
        else:
            assessment["protein"] = "adequate"
        
        return assessment
    
    def _estimate_cost(self, shopping_list: Dict[str, List[Dict[str, Any]]], budget_range: Optional[str]) -> float:
        """Estimate the cost of the shopping list."""
        # Simplified cost estimation
        base_costs = {
            "produce": 2.0,
            "meat_seafood": 8.0,
            "dairy": 3.0,
            "pantry": 1.5,
            "spices": 1.0,
            "other": 2.5
        }
        
        total_cost = 0
        for category, items in shopping_list.items():
            category_cost = base_costs.get(category, 2.0)
            total_cost += len(items) * category_cost
        
        # Adjust based on budget range
        if budget_range == "low":
            total_cost *= 0.7
        elif budget_range == "high":
            total_cost *= 1.5
        
        return round(total_cost, 2)
    
    async def generate_shopping_list(self, plan_id: int, consolidate_by_store: bool = False) -> Dict[str, Any]:
        """Generate shopping list for an existing meal plan."""
        # Implementation would fetch meal plan from database
        # For now, return placeholder
        return {"message": "Shopping list generated", "plan_id": plan_id}
    
    async def substitute_meal(self, plan_id: int, meal_id: int, dietary_restrictions: Optional[List[str]], cuisine_preference: Optional[str]) -> Dict[str, Any]:
        """Substitute a meal in an existing plan."""
        # Implementation would fetch plan, generate new meal, and update
        return {"message": "Meal substituted", "plan_id": plan_id, "meal_id": meal_id}
    
    async def analyze_nutrition(self, plan_id: int, target_calories: Optional[int], target_protein: Optional[int]) -> Dict[str, Any]:
        """Analyze nutrition for an existing meal plan."""
        # Implementation would fetch plan and analyze
        return {"message": "Nutrition analyzed", "plan_id": plan_id}
    
    async def optimize_plan(self, plan_id: int, optimization_goals: List[str], constraints: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Optimize an existing meal plan."""
        # Implementation would apply optimization algorithms
        return {"message": "Plan optimized", "plan_id": plan_id, "goals": optimization_goals}