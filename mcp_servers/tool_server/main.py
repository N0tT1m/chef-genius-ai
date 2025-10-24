"""
Tool Integration MCP Server
Handles nutrition analysis, vision processing, and utility tools
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path
import base64
import io
from datetime import datetime, timedelta

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.types import Tool, TextContent
from backend.app.services.enhanced_rag_system import EnhancedRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ToolIntegrationServer:
    """MCP Server for tool integration and utility functions."""
    
    def __init__(self, rag_system: EnhancedRAGSystem = None):
        """
        Initialize the tool integration server.
        
        Args:
            rag_system: Enhanced RAG system instance
        """
        self.server = Server("tool-integration")
        self.rag_system = rag_system or EnhancedRAGSystem()
        
        # Nutrition database (simplified)
        self.nutrition_db = self._load_nutrition_database()
        
        # Shopping optimization data
        self.store_data = self._load_store_data()
        
        # Performance tracking
        self.tool_usage_count = 0
        self.nutrition_requests = 0
        self.shopping_lists_generated = 0
        
        self._setup_tools()
    
    def _load_nutrition_database(self) -> Dict[str, Dict]:
        """Load nutrition database for ingredients."""
        return {
            "chicken breast": {
                "calories_per_100g": 165,
                "protein": 31,
                "carbs": 0,
                "fat": 3.6,
                "fiber": 0,
                "nutrients": {"vitamin_b6": "high", "niacin": "high", "phosphorus": "high"}
            },
            "rice": {
                "calories_per_100g": 130,
                "protein": 2.7,
                "carbs": 28,
                "fat": 0.3,
                "fiber": 0.4,
                "nutrients": {"manganese": "high", "selenium": "moderate"}
            },
            "broccoli": {
                "calories_per_100g": 34,
                "protein": 2.8,
                "carbs": 7,
                "fat": 0.4,
                "fiber": 2.6,
                "nutrients": {"vitamin_c": "very_high", "vitamin_k": "very_high", "folate": "high"}
            },
            "olive oil": {
                "calories_per_100g": 884,
                "protein": 0,
                "carbs": 0,
                "fat": 100,
                "fiber": 0,
                "nutrients": {"vitamin_e": "high", "vitamin_k": "moderate"}
            },
            "tomato": {
                "calories_per_100g": 18,
                "protein": 0.9,
                "carbs": 3.9,
                "fat": 0.2,
                "fiber": 1.2,
                "nutrients": {"vitamin_c": "high", "lycopene": "high", "potassium": "moderate"}
            }
        }
    
    def _load_store_data(self) -> Dict[str, Any]:
        """Load store and pricing data."""
        return {
            "store_types": {
                "supermarket": {"cost_multiplier": 1.0, "availability": 0.9},
                "organic_store": {"cost_multiplier": 1.4, "availability": 0.7},
                "discount_store": {"cost_multiplier": 0.8, "availability": 0.6},
                "farmers_market": {"cost_multiplier": 1.2, "availability": 0.5}
            },
            "seasonal_pricing": {
                "spring": ["asparagus", "peas", "strawberries"],
                "summer": ["tomatoes", "corn", "berries"],
                "fall": ["apples", "pumpkin", "squash"],
                "winter": ["citrus", "root vegetables", "cabbage"]
            }
        }
    
    def _setup_tools(self):
        """Setup MCP tools for various integrations."""
        
        @self.server.tool("analyze_nutrition")
        async def analyze_nutrition(
            recipe: str,
            servings: int = 4,
            detailed: bool = False
        ) -> Dict[str, Any]:
            """
            Analyze nutritional content of a recipe.
            
            Args:
                recipe: Recipe text to analyze
                servings: Number of servings
                detailed: Whether to include detailed nutrient breakdown
                
            Returns:
                Nutritional analysis
            """
            return await self._analyze_nutrition(recipe, servings, detailed)
        
        @self.server.tool("generate_shopping_list")
        async def generate_shopping_list(
            recipes: List[str],
            servings: List[int],
            dietary_preferences: Optional[List[str]] = None,
            budget_target: Optional[float] = None
        ) -> Dict[str, Any]:
            """
            Generate optimized shopping list for multiple recipes.
            
            Args:
                recipes: List of recipe texts or titles
                servings: Number of servings for each recipe
                dietary_preferences: Dietary preferences to consider
                budget_target: Target budget for shopping
                
            Returns:
                Optimized shopping list with cost estimates
            """
            return await self._generate_shopping_list(recipes, servings, dietary_preferences, budget_target)
        
        @self.server.tool("analyze_food_image")
        async def analyze_food_image(
            image_data: str,
            image_format: str = "base64"
        ) -> Dict[str, Any]:
            """
            Analyze food image to identify ingredients and suggest recipes.
            
            Args:
                image_data: Base64 encoded image data
                image_format: Format of the image data
                
            Returns:
                Image analysis with ingredient identification
            """
            return await self._analyze_food_image(image_data, image_format)
        
        @self.server.tool("plan_meal_prep")
        async def plan_meal_prep(
            dietary_goals: Dict[str, Any],
            days: int = 7,
            meals_per_day: int = 3,
            preferences: Optional[Dict[str, Any]] = None
        ) -> Dict[str, Any]:
            """
            Generate meal prep plan based on dietary goals.
            
            Args:
                dietary_goals: Target nutrition and dietary requirements
                days: Number of days to plan for
                meals_per_day: Number of meals per day
                preferences: User preferences and restrictions
                
            Returns:
                Complete meal prep plan with shopping and prep schedule
            """
            return await self._plan_meal_prep(dietary_goals, days, meals_per_day, preferences)
        
        @self.server.tool("estimate_cooking_time")
        async def estimate_cooking_time(
            recipe: str,
            skill_level: str = "intermediate",
            equipment_available: Optional[List[str]] = None
        ) -> Dict[str, Any]:
            """
            Estimate cooking time based on recipe and user factors.
            
            Args:
                recipe: Recipe text to analyze
                skill_level: User's cooking skill level
                equipment_available: Available cooking equipment
                
            Returns:
                Time estimates for preparation and cooking
            """
            return await self._estimate_cooking_time(recipe, skill_level, equipment_available)
        
        @self.server.tool("suggest_wine_pairing")
        async def suggest_wine_pairing(
            recipe: str,
            occasion: Optional[str] = None,
            budget_range: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Suggest wine pairings for a recipe.
            
            Args:
                recipe: Recipe text to pair with
                occasion: Type of occasion (casual, formal, etc.)
                budget_range: Budget range for wine selection
                
            Returns:
                Wine pairing suggestions with rationale
            """
            return await self._suggest_wine_pairing(recipe, occasion, budget_range)
        
        @self.server.tool("calculate_recipe_scaling")
        async def calculate_recipe_scaling(
            recipe: str,
            original_servings: int,
            target_servings: int
        ) -> Dict[str, Any]:
            """
            Scale recipe ingredients for different serving sizes.
            
            Args:
                recipe: Original recipe text
                original_servings: Original number of servings
                target_servings: Target number of servings
                
            Returns:
                Scaled recipe with adjusted ingredients
            """
            return await self._calculate_recipe_scaling(recipe, original_servings, target_servings)
        
        @self.server.tool("check_food_safety")
        async def check_food_safety(
            ingredients: List[str],
            storage_method: str,
            storage_duration: int
        ) -> Dict[str, Any]:
            """
            Check food safety for ingredient storage and handling.
            
            Args:
                ingredients: List of ingredients to check
                storage_method: Storage method (refrigerated, frozen, pantry)
                storage_duration: Storage duration in days
                
            Returns:
                Food safety assessment and recommendations
            """
            return await self._check_food_safety(ingredients, storage_method, storage_duration)
    
    async def _analyze_nutrition(self, recipe: str, servings: int, detailed: bool) -> Dict[str, Any]:
        """Analyze nutritional content of a recipe."""
        self.nutrition_requests += 1
        
        try:
            # Extract ingredients from recipe
            ingredients = self._extract_ingredients_from_recipe(recipe)
            
            # Calculate nutrition totals
            total_nutrition = {
                "calories": 0,
                "protein": 0,
                "carbs": 0,
                "fat": 0,
                "fiber": 0
            }
            
            ingredient_nutrition = {}
            
            for ingredient_info in ingredients:
                ingredient_name = ingredient_info["name"]
                amount = ingredient_info["amount"]
                
                # Look up nutrition data
                nutrition_data = self._get_ingredient_nutrition(ingredient_name)
                if nutrition_data:
                    # Scale by amount
                    scaled_nutrition = self._scale_nutrition(nutrition_data, amount)
                    
                    # Add to totals
                    for key in total_nutrition:
                        total_nutrition[key] += scaled_nutrition.get(key, 0)
                    
                    if detailed:
                        ingredient_nutrition[ingredient_name] = scaled_nutrition
            
            # Calculate per serving
            per_serving = {}
            for key, value in total_nutrition.items():
                per_serving[key] = round(value / servings, 1)
            
            # Nutritional assessment
            assessment = self._assess_nutrition(per_serving)
            
            result = {
                "total_nutrition": total_nutrition,
                "per_serving": per_serving,
                "servings": servings,
                "assessment": assessment
            }
            
            if detailed:
                result["ingredient_breakdown"] = ingredient_nutrition
                result["daily_value_percentages"] = self._calculate_daily_values(per_serving)
            
            return result
            
        except Exception as e:
            logger.error(f"Nutrition analysis failed: {e}")
            return {"error": str(e)}
    
    def _extract_ingredients_from_recipe(self, recipe: str) -> List[Dict[str, Any]]:
        """Extract ingredients from recipe text."""
        # Simple ingredient extraction (could be enhanced with NLP)
        ingredients = []
        lines = recipe.split('\n')
        
        in_ingredients_section = False
        for line in lines:
            line = line.strip()
            
            if 'ingredients' in line.lower():
                in_ingredients_section = True
                continue
            
            if in_ingredients_section:
                if line and not line.startswith('instructions') and not line.startswith('method'):
                    # Parse ingredient line
                    ingredient_info = self._parse_ingredient_line(line)
                    if ingredient_info:
                        ingredients.append(ingredient_info)
                else:
                    in_ingredients_section = False
        
        return ingredients
    
    def _parse_ingredient_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a single ingredient line."""
        # Simple parsing - could be enhanced with more sophisticated NLP
        import re
        
        # Remove common formatting
        line = re.sub(r'^[-*•]\s*', '', line)
        
        # Try to extract amount and unit
        amount_match = re.match(r'(\d+(?:\.\d+)?(?:/\d+)?)\s*(\w+)?\s*(.*)', line)
        
        if amount_match:
            amount_str = amount_match.group(1)
            unit = amount_match.group(2) or "piece"
            ingredient_name = amount_match.group(3).strip()
            
            # Convert amount to grams (simplified)
            amount_grams = self._convert_to_grams(amount_str, unit, ingredient_name)
            
            return {
                "name": ingredient_name,
                "amount": amount_grams,
                "original_line": line
            }
        
        # If no amount found, assume 100g
        return {
            "name": line,
            "amount": 100,
            "original_line": line
        }
    
    def _convert_to_grams(self, amount_str: str, unit: str, ingredient: str) -> float:
        """Convert ingredient amount to grams."""
        try:
            # Handle fractions
            if '/' in amount_str:
                parts = amount_str.split('/')
                amount = float(parts[0]) / float(parts[1])
            else:
                amount = float(amount_str)
            
            # Convert based on unit
            unit_conversions = {
                "cup": 240,  # ml, varies by ingredient
                "tbsp": 15,
                "tsp": 5,
                "oz": 28.35,
                "lb": 453.6,
                "kg": 1000,
                "g": 1,
                "ml": 1,  # Assume 1ml = 1g for simplicity
                "l": 1000
            }
            
            if unit.lower() in unit_conversions:
                return amount * unit_conversions[unit.lower()]
            
            # Default to the amount as grams
            return amount * 100  # Assume 100g per unit if unknown
            
        except ValueError:
            return 100  # Default amount
    
    def _get_ingredient_nutrition(self, ingredient_name: str) -> Optional[Dict[str, Any]]:
        """Get nutrition data for an ingredient."""
        ingredient_lower = ingredient_name.lower()
        
        # Direct lookup
        if ingredient_lower in self.nutrition_db:
            return self.nutrition_db[ingredient_lower]
        
        # Partial matching
        for key in self.nutrition_db:
            if key in ingredient_lower or ingredient_lower in key:
                return self.nutrition_db[key]
        
        # Return default nutrition for unknown ingredients
        return {
            "calories_per_100g": 50,
            "protein": 2,
            "carbs": 10,
            "fat": 1,
            "fiber": 1,
            "nutrients": {}
        }
    
    def _scale_nutrition(self, nutrition_data: Dict, amount_grams: float) -> Dict[str, float]:
        """Scale nutrition data by ingredient amount."""
        scale_factor = amount_grams / 100  # Nutrition data is per 100g
        
        return {
            "calories": nutrition_data.get("calories_per_100g", 0) * scale_factor,
            "protein": nutrition_data.get("protein", 0) * scale_factor,
            "carbs": nutrition_data.get("carbs", 0) * scale_factor,
            "fat": nutrition_data.get("fat", 0) * scale_factor,
            "fiber": nutrition_data.get("fiber", 0) * scale_factor
        }
    
    def _assess_nutrition(self, per_serving: Dict[str, float]) -> Dict[str, Any]:
        """Assess nutritional balance of a recipe."""
        assessment = {
            "calorie_level": "moderate",
            "protein_adequate": True,
            "balance_score": 0.7,
            "recommendations": []
        }
        
        calories = per_serving.get("calories", 0)
        protein = per_serving.get("protein", 0)
        carbs = per_serving.get("carbs", 0)
        fat = per_serving.get("fat", 0)
        
        # Calorie assessment
        if calories < 200:
            assessment["calorie_level"] = "low"
        elif calories > 600:
            assessment["calorie_level"] = "high"
        
        # Protein assessment
        if protein < 10:
            assessment["protein_adequate"] = False
            assessment["recommendations"].append("Consider adding more protein sources")
        
        # Balance assessment
        total_macros = protein + carbs + fat
        if total_macros > 0:
            protein_pct = (protein * 4) / (calories) * 100 if calories > 0 else 0
            carb_pct = (carbs * 4) / (calories) * 100 if calories > 0 else 0
            fat_pct = (fat * 9) / (calories) * 100 if calories > 0 else 0
            
            # Check for balanced macros
            if protein_pct < 15:
                assessment["recommendations"].append("Consider increasing protein content")
            if carb_pct > 60:
                assessment["recommendations"].append("Consider reducing carbohydrate content")
            if fat_pct > 35:
                assessment["recommendations"].append("Consider reducing fat content")
        
        return assessment
    
    def _calculate_daily_values(self, per_serving: Dict[str, float]) -> Dict[str, float]:
        """Calculate percentage daily values."""
        daily_values = {
            "calories": 2000,
            "protein": 50,
            "carbs": 300,
            "fat": 65,
            "fiber": 25
        }
        
        percentages = {}
        for nutrient, amount in per_serving.items():
            if nutrient in daily_values:
                percentages[nutrient] = round((amount / daily_values[nutrient]) * 100, 1)
        
        return percentages
    
    async def _generate_shopping_list(self, recipes: List[str], servings: List[int], 
                                    dietary_preferences: Optional[List[str]], 
                                    budget_target: Optional[float]) -> Dict[str, Any]:
        """Generate optimized shopping list."""
        self.shopping_lists_generated += 1
        
        try:
            all_ingredients = {}
            
            # Extract ingredients from all recipes
            for i, recipe in enumerate(recipes):
                recipe_servings = servings[i] if i < len(servings) else 4
                ingredients = self._extract_ingredients_from_recipe(recipe)
                
                for ingredient_info in ingredients:
                    name = ingredient_info["name"]
                    amount = ingredient_info["amount"] * (recipe_servings / 4)  # Scale by servings
                    
                    if name in all_ingredients:
                        all_ingredients[name]["total_amount"] += amount
                        all_ingredients[name]["used_in"].append(f"Recipe {i+1}")
                    else:
                        all_ingredients[name] = {
                            "total_amount": amount,
                            "unit": "g",
                            "used_in": [f"Recipe {i+1}"],
                            "category": self._categorize_ingredient(name)
                        }
            
            # Optimize quantities
            optimized_ingredients = self._optimize_quantities(all_ingredients)
            
            # Estimate costs
            cost_estimates = self._estimate_costs(optimized_ingredients)
            
            # Suggest stores
            store_suggestions = self._suggest_stores(optimized_ingredients, dietary_preferences, budget_target)
            
            # Organize by category
            organized_list = self._organize_by_category(optimized_ingredients)
            
            return {
                "shopping_list": organized_list,
                "total_estimated_cost": sum(cost_estimates.values()),
                "cost_breakdown": cost_estimates,
                "store_suggestions": store_suggestions,
                "optimization_notes": self._get_optimization_notes(all_ingredients, optimized_ingredients)
            }
            
        except Exception as e:
            logger.error(f"Shopping list generation failed: {e}")
            return {"error": str(e)}
    
    def _categorize_ingredient(self, ingredient_name: str) -> str:
        """Categorize ingredient for shopping organization."""
        ingredient_lower = ingredient_name.lower()
        
        categories = {
            "produce": ["tomato", "onion", "carrot", "lettuce", "apple", "banana", "broccoli"],
            "meat": ["chicken", "beef", "pork", "fish", "turkey"],
            "dairy": ["milk", "cheese", "yogurt", "butter", "cream"],
            "pantry": ["rice", "pasta", "oil", "flour", "sugar", "salt"],
            "spices": ["pepper", "cumin", "oregano", "basil", "paprika"],
            "frozen": ["frozen vegetables", "ice cream"],
            "bakery": ["bread", "rolls", "bagels"]
        }
        
        for category, items in categories.items():
            if any(item in ingredient_lower for item in items):
                return category
        
        return "miscellaneous"
    
    def _optimize_quantities(self, ingredients: Dict[str, Dict]) -> Dict[str, Dict]:
        """Optimize ingredient quantities for practical shopping."""
        optimized = {}
        
        for name, info in ingredients.items():
            total_amount = info["total_amount"]
            
            # Round up to practical quantities
            if total_amount < 50:  # Less than 50g
                practical_amount = 100  # Buy smallest package
            elif total_amount < 250:
                practical_amount = 250
            elif total_amount < 500:
                practical_amount = 500
            else:
                practical_amount = round(total_amount / 100) * 100  # Round to nearest 100g
            
            optimized[name] = info.copy()
            optimized[name]["practical_amount"] = practical_amount
            optimized[name]["waste_estimate"] = practical_amount - total_amount
        
        return optimized
    
    def _estimate_costs(self, ingredients: Dict[str, Dict]) -> Dict[str, float]:
        """Estimate costs for ingredients."""
        # Simple cost estimation (could be enhanced with real pricing data)
        base_costs_per_kg = {
            "produce": 3.0,
            "meat": 12.0,
            "dairy": 4.0,
            "pantry": 2.0,
            "spices": 20.0,
            "frozen": 3.5,
            "bakery": 2.5,
            "miscellaneous": 5.0
        }
        
        costs = {}
        for name, info in ingredients.items():
            category = info["category"]
            amount_kg = info["practical_amount"] / 1000
            base_cost = base_costs_per_kg.get(category, 5.0)
            costs[name] = round(amount_kg * base_cost, 2)
        
        return costs
    
    def _suggest_stores(self, ingredients: Dict[str, Dict], 
                       dietary_preferences: Optional[List[str]], 
                       budget_target: Optional[float]) -> List[Dict[str, Any]]:
        """Suggest optimal stores for shopping."""
        suggestions = []
        
        # Analyze ingredient categories
        categories = set(info["category"] for info in ingredients.values())
        
        # General supermarket
        suggestions.append({
            "store_type": "supermarket",
            "pros": ["One-stop shopping", "Good availability"],
            "cons": ["Average prices"],
            "recommended_for": list(categories)
        })
        
        # Specialty suggestions based on dietary preferences
        if dietary_preferences:
            if "organic" in dietary_preferences:
                suggestions.append({
                    "store_type": "organic_store",
                    "pros": ["Organic options", "High quality"],
                    "cons": ["Higher prices"],
                    "recommended_for": ["produce", "dairy"]
                })
        
        # Budget considerations
        if budget_target and budget_target < 50:
            suggestions.append({
                "store_type": "discount_store",
                "pros": ["Lower prices", "Good for bulk items"],
                "cons": ["Limited selection"],
                "recommended_for": ["pantry", "frozen"]
            })
        
        return suggestions
    
    def _organize_by_category(self, ingredients: Dict[str, Dict]) -> Dict[str, List[Dict]]:
        """Organize shopping list by category."""
        organized = {}
        
        for name, info in ingredients.items():
            category = info["category"]
            if category not in organized:
                organized[category] = []
            
            organized[category].append({
                "name": name,
                "amount": info["practical_amount"],
                "unit": info["unit"],
                "used_in": info["used_in"]
            })
        
        return organized
    
    def _get_optimization_notes(self, original: Dict, optimized: Dict) -> List[str]:
        """Get notes about shopping list optimization."""
        notes = []
        
        total_waste = sum(info["waste_estimate"] for info in optimized.values())
        if total_waste > 100:
            notes.append(f"Optimized quantities may result in ~{int(total_waste)}g of extra ingredients")
        
        bulk_items = [name for name, info in optimized.items() if info["practical_amount"] > 500]
        if bulk_items:
            notes.append(f"Consider buying in bulk: {', '.join(bulk_items[:3])}")
        
        return notes
    
    async def _analyze_food_image(self, image_data: str, image_format: str) -> Dict[str, Any]:
        """Analyze food image (simplified mock implementation)."""
        try:
            # Mock image analysis - in real implementation, this would use computer vision
            mock_analysis = {
                "identified_ingredients": [
                    {"name": "tomato", "confidence": 0.95},
                    {"name": "basil", "confidence": 0.87},
                    {"name": "cheese", "confidence": 0.92}
                ],
                "dish_type": "pasta dish",
                "cuisine_style": "Italian",
                "confidence_score": 0.91
            }
            
            # Find recipes using identified ingredients
            ingredient_names = [ing["name"] for ing in mock_analysis["identified_ingredients"]]
            recipe_suggestions = await self.rag_system.hybrid_search(
                query=" ".join(ingredient_names),
                top_k=3
            )
            
            return {
                "analysis": mock_analysis,
                "recipe_suggestions": recipe_suggestions,
                "processing_time": "1.2s"
            }
            
        except Exception as e:
            logger.error(f"Image analysis failed: {e}")
            return {"error": str(e)}
    
    async def _plan_meal_prep(self, dietary_goals: Dict[str, Any], days: int, 
                            meals_per_day: int, preferences: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate meal prep plan."""
        try:
            # Extract goals
            target_calories = dietary_goals.get("daily_calories", 2000)
            target_protein = dietary_goals.get("daily_protein", 150)
            dietary_restrictions = dietary_goals.get("restrictions", [])
            
            # Calculate per-meal targets
            calories_per_meal = target_calories / meals_per_day
            protein_per_meal = target_protein / meals_per_day
            
            # Generate meal plan
            meal_plan = []
            
            for day in range(days):
                day_meals = []
                
                for meal_num in range(meals_per_day):
                    # Search for appropriate recipes
                    query = f"healthy {calories_per_meal} calories"
                    if dietary_restrictions:
                        query += f" {' '.join(dietary_restrictions)}"
                    
                    recipes = await self.rag_system.hybrid_search(
                        query=query,
                        top_k=3,
                        filters={"dietary_restrictions": dietary_restrictions}
                    )
                    
                    if recipes:
                        selected_recipe = recipes[0]  # Select top recipe
                        day_meals.append({
                            "meal_number": meal_num + 1,
                            "recipe": selected_recipe,
                            "target_calories": calories_per_meal,
                            "target_protein": protein_per_meal
                        })
                
                meal_plan.append({
                    "day": day + 1,
                    "meals": day_meals
                })
            
            # Generate prep schedule
            prep_schedule = self._create_prep_schedule(meal_plan)
            
            # Generate shopping list
            all_recipes = []
            all_servings = []
            for day in meal_plan:
                for meal in day["meals"]:
                    all_recipes.append(meal["recipe"].get("title", "Unknown Recipe"))
                    all_servings.append(1)
            
            shopping_list = await self._generate_shopping_list(
                all_recipes, all_servings, dietary_restrictions
            )
            
            return {
                "meal_plan": meal_plan,
                "prep_schedule": prep_schedule,
                "shopping_list": shopping_list,
                "nutritional_summary": self._summarize_meal_plan_nutrition(meal_plan),
                "total_days": days,
                "meals_per_day": meals_per_day
            }
            
        except Exception as e:
            logger.error(f"Meal prep planning failed: {e}")
            return {"error": str(e)}
    
    def _create_prep_schedule(self, meal_plan: List[Dict]) -> Dict[str, List[str]]:
        """Create meal prep schedule."""
        schedule = {
            "day_before": [
                "Shop for all ingredients",
                "Prep non-perishable ingredients"
            ],
            "prep_day": [
                "Batch cook proteins",
                "Prepare base grains and starches",
                "Wash and prep vegetables",
                "Portion meals into containers"
            ],
            "daily_tasks": [
                "Add fresh elements (herbs, dressings)",
                "Reheat as needed"
            ]
        }
        
        return schedule
    
    def _summarize_meal_plan_nutrition(self, meal_plan: List[Dict]) -> Dict[str, Any]:
        """Summarize nutrition across the meal plan."""
        return {
            "average_daily_calories": 2000,  # Mock calculation
            "average_daily_protein": 150,
            "variety_score": 0.8,
            "balance_assessment": "Well-balanced with good variety"
        }
    
    async def _estimate_cooking_time(self, recipe: str, skill_level: str, 
                                   equipment_available: Optional[List[str]]) -> Dict[str, Any]:
        """Estimate cooking time for recipe."""
        try:
            # Base time estimation (simplified)
            base_prep_time = 15  # minutes
            base_cook_time = 30
            
            # Adjust for skill level
            skill_multipliers = {
                "beginner": 1.5,
                "intermediate": 1.0,
                "advanced": 0.8
            }
            
            multiplier = skill_multipliers.get(skill_level, 1.0)
            
            # Analyze recipe complexity
            complexity_indicators = ["marinade", "multiple steps", "sauce", "pastry"]
            complexity_count = sum(1 for indicator in complexity_indicators if indicator in recipe.lower())
            
            # Adjust times
            prep_time = int(base_prep_time * multiplier * (1 + complexity_count * 0.2))
            cook_time = int(base_cook_time * multiplier)
            
            # Equipment adjustments
            if equipment_available:
                if "food processor" in equipment_available:
                    prep_time = int(prep_time * 0.8)  # Faster prep
                if "pressure cooker" in equipment_available:
                    cook_time = int(cook_time * 0.6)  # Faster cooking
            
            return {
                "estimated_prep_time": prep_time,
                "estimated_cook_time": cook_time,
                "total_time": prep_time + cook_time,
                "skill_level": skill_level,
                "complexity_factors": complexity_indicators[:complexity_count],
                "time_breakdown": {
                    "preparation": prep_time,
                    "cooking": cook_time,
                    "equipment_bonus": "Yes" if equipment_available else "No"
                }
            }
            
        except Exception as e:
            logger.error(f"Cooking time estimation failed: {e}")
            return {"error": str(e)}
    
    async def _suggest_wine_pairing(self, recipe: str, occasion: Optional[str], 
                                  budget_range: Optional[str]) -> Dict[str, Any]:
        """Suggest wine pairings."""
        try:
            # Simple wine pairing logic (could be enhanced with wine database)
            recipe_lower = recipe.lower()
            
            pairings = []
            
            # Protein-based pairings
            if any(protein in recipe_lower for protein in ["chicken", "poultry"]):
                pairings.append({
                    "wine_type": "Chardonnay",
                    "style": "White",
                    "rationale": "Complements poultry with balanced acidity"
                })
            
            if any(protein in recipe_lower for protein in ["beef", "steak"]):
                pairings.append({
                    "wine_type": "Cabernet Sauvignon",
                    "style": "Red",
                    "rationale": "Bold flavors complement red meat"
                })
            
            if any(protein in recipe_lower for protein in ["fish", "seafood"]):
                pairings.append({
                    "wine_type": "Sauvignon Blanc",
                    "style": "White",
                    "rationale": "Crisp acidity enhances seafood flavors"
                })
            
            # Cuisine-based pairings
            if "italian" in recipe_lower or "pasta" in recipe_lower:
                pairings.append({
                    "wine_type": "Chianti",
                    "style": "Red",
                    "rationale": "Traditional Italian pairing"
                })
            
            # Default pairing
            if not pairings:
                pairings.append({
                    "wine_type": "Pinot Grigio",
                    "style": "White",
                    "rationale": "Versatile wine that pairs with many dishes"
                })
            
            # Add budget considerations
            budget_suggestions = []
            if budget_range == "budget":
                budget_suggestions.append("Look for regional wines or house wines")
            elif budget_range == "premium":
                budget_suggestions.append("Consider reserve or vintage selections")
            
            return {
                "wine_pairings": pairings,
                "occasion_notes": f"For {occasion} occasions" if occasion else "For any occasion",
                "budget_suggestions": budget_suggestions,
                "serving_suggestions": "Serve at appropriate temperature and decant if needed"
            }
            
        except Exception as e:
            logger.error(f"Wine pairing suggestion failed: {e}")
            return {"error": str(e)}
    
    async def _calculate_recipe_scaling(self, recipe: str, original_servings: int, 
                                      target_servings: int) -> Dict[str, Any]:
        """Scale recipe for different serving sizes."""
        try:
            scale_factor = target_servings / original_servings
            
            # Extract ingredients
            ingredients = self._extract_ingredients_from_recipe(recipe)
            
            # Scale ingredients
            scaled_ingredients = []
            for ingredient in ingredients:
                original_amount = ingredient["amount"]
                scaled_amount = original_amount * scale_factor
                
                scaled_ingredients.append({
                    "name": ingredient["name"],
                    "original_amount": original_amount,
                    "scaled_amount": round(scaled_amount, 1),
                    "scale_factor": scale_factor
                })
            
            # Scaling considerations
            considerations = []
            if scale_factor > 2:
                considerations.append("Large increases may require longer cooking times")
                considerations.append("Consider cooking in batches")
            elif scale_factor < 0.5:
                considerations.append("Small portions may cook faster")
                considerations.append("Watch cooking times carefully")
            
            return {
                "original_servings": original_servings,
                "target_servings": target_servings,
                "scale_factor": round(scale_factor, 2),
                "scaled_ingredients": scaled_ingredients,
                "scaling_considerations": considerations,
                "cooking_time_adjustment": self._estimate_time_adjustment(scale_factor)
            }
            
        except Exception as e:
            logger.error(f"Recipe scaling failed: {e}")
            return {"error": str(e)}
    
    def _estimate_time_adjustment(self, scale_factor: float) -> str:
        """Estimate cooking time adjustment for scaling."""
        if scale_factor > 2:
            return "Increase cooking time by 20-30%"
        elif scale_factor < 0.5:
            return "Reduce cooking time by 15-25%"
        else:
            return "Minimal time adjustment needed"
    
    async def _check_food_safety(self, ingredients: List[str], storage_method: str, 
                               storage_duration: int) -> Dict[str, Any]:
        """Check food safety for ingredients."""
        try:
            safety_results = {}
            
            # Food safety database (simplified)
            safety_limits = {
                "refrigerated": {
                    "meat": 3,
                    "poultry": 2,
                    "fish": 1,
                    "dairy": 7,
                    "vegetables": 7,
                    "cooked_food": 4
                },
                "frozen": {
                    "meat": 180,
                    "poultry": 120,
                    "fish": 90,
                    "vegetables": 365
                },
                "pantry": {
                    "dry_goods": 365,
                    "canned": 730,
                    "spices": 1095
                }
            }
            
            storage_limits = safety_limits.get(storage_method, {})
            
            for ingredient in ingredients:
                ingredient_category = self._categorize_for_safety(ingredient)
                safe_days = storage_limits.get(ingredient_category, 1)
                
                is_safe = storage_duration <= safe_days
                
                safety_results[ingredient] = {
                    "safe": is_safe,
                    "max_safe_days": safe_days,
                    "current_days": storage_duration,
                    "category": ingredient_category,
                    "recommendation": "Safe to use" if is_safe else "Discard - exceeded safe storage time"
                }
            
            # Overall assessment
            all_safe = all(result["safe"] for result in safety_results.values())
            
            return {
                "overall_safe": all_safe,
                "ingredient_safety": safety_results,
                "storage_method": storage_method,
                "storage_duration": storage_duration,
                "general_recommendations": [
                    "When in doubt, throw it out",
                    "Check for signs of spoilage",
                    "Follow first-in-first-out principle"
                ]
            }
            
        except Exception as e:
            logger.error(f"Food safety check failed: {e}")
            return {"error": str(e)}
    
    def _categorize_for_safety(self, ingredient: str) -> str:
        """Categorize ingredient for food safety purposes."""
        ingredient_lower = ingredient.lower()
        
        if any(meat in ingredient_lower for meat in ["beef", "pork", "lamb"]):
            return "meat"
        elif any(poultry in ingredient_lower for poultry in ["chicken", "turkey", "duck"]):
            return "poultry"
        elif any(fish in ingredient_lower for fish in ["fish", "salmon", "tuna", "shrimp"]):
            return "fish"
        elif any(dairy in ingredient_lower for dairy in ["milk", "cheese", "yogurt", "cream"]):
            return "dairy"
        elif any(veg in ingredient_lower for veg in ["lettuce", "spinach", "tomato", "carrot"]):
            return "vegetables"
        elif any(dry in ingredient_lower for dry in ["rice", "pasta", "flour", "sugar"]):
            return "dry_goods"
        else:
            return "cooked_food"  # Default to most restrictive
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        return {
            "server_name": "tool-integration",
            "tool_usage_count": self.tool_usage_count,
            "nutrition_requests": self.nutrition_requests,
            "shopping_lists_generated": self.shopping_lists_generated,
            "nutrition_db_entries": len(self.nutrition_db),
            "store_types": len(self.store_data["store_types"])
        }

# Server startup
async def main():
    """Start the Tool Integration MCP Server."""
    try:
        # Initialize RAG system
        rag_system = EnhancedRAGSystem()
        
        # Initialize server
        server = ToolIntegrationServer(rag_system=rag_system)
        
        # Start server
        logger.info("Starting Tool Integration MCP Server...")
        await server.server.run()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())