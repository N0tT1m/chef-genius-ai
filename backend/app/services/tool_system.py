"""
Tool System for Chef Genius

This module implements a sophisticated tool calling system that allows AI models
to use external functions for calculations, data retrieval, and actions.
"""

import logging
import asyncio
import json
import math
from typing import List, Dict, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from abc import ABC, abstractmethod
import requests
from app.services.nutrition_analyzer import NutritionAnalyzer
from app.services.substitution_engine import SubstitutionEngine

logger = logging.getLogger(__name__)

@dataclass
class ToolCall:
    """Represents a tool call request."""
    name: str
    arguments: Dict[str, Any]
    call_id: str = ""

@dataclass
class ToolResult:
    """Represents the result of a tool call."""
    call_id: str
    name: str
    result: Any
    success: bool
    error: Optional[str] = None

class BaseTool(ABC):
    """Base class for all tools."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name identifier."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for AI model."""
        pass
    
    @property
    @abstractmethod
    def parameters(self) -> Dict[str, Any]:
        """Tool parameters schema."""
        pass
    
    @abstractmethod
    async def call(self, **kwargs) -> Any:
        """Execute the tool."""
        pass

class NutritionCalculatorTool(BaseTool):
    """Tool for calculating nutritional information."""
    
    def __init__(self):
        self.nutrition_analyzer = NutritionAnalyzer()
    
    @property
    def name(self) -> str:
        return "nutrition_calculator"
    
    @property
    def description(self) -> str:
        return "Calculate nutritional information for ingredients or recipes including calories, macros, and vitamins."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "amount": {"type": "number"},
                            "unit": {"type": "string"}
                        },
                        "required": ["name", "amount", "unit"]
                    },
                    "description": "List of ingredients with amounts"
                },
                "servings": {
                    "type": "integer",
                    "description": "Number of servings",
                    "default": 1
                }
            },
            "required": ["ingredients"]
        }
    
    async def call(self, ingredients: List[Dict], servings: int = 1) -> Dict[str, Any]:
        """Calculate nutrition for given ingredients."""
        try:
            # Convert to format expected by nutrition analyzer
            nutrition_request = {
                "ingredients": ingredients,
                "servings": servings
            }
            
            result = await self.nutrition_analyzer.analyze_recipe_nutrition(nutrition_request)
            return {
                "success": True,
                "nutrition": result,
                "per_serving": True if servings > 1 else False
            }
        except Exception as e:
            logger.error(f"Nutrition calculation failed: {e}")
            return {"success": False, "error": str(e)}

class UnitConversionTool(BaseTool):
    """Tool for converting cooking units and measurements."""
    
    # Conversion factors to grams/ml
    UNIT_CONVERSIONS = {
        # Weight conversions (to grams)
        "kg": 1000, "g": 1, "mg": 0.001,
        "lb": 453.592, "oz": 28.3495,
        # Volume conversions (to ml)
        "l": 1000, "ml": 1, "dl": 100, "cl": 10,
        "cup": 240, "tbsp": 15, "tsp": 5,
        "fl oz": 29.5735, "pint": 473.176, "quart": 946.353, "gallon": 3785.41,
        # Common cooking conversions
        "stick_butter": 113.4,  # 1 stick = 4 oz = 113.4g
    }
    
    # Temperature conversions
    @staticmethod
    def celsius_to_fahrenheit(celsius: float) -> float:
        return (celsius * 9/5) + 32
    
    @staticmethod
    def fahrenheit_to_celsius(fahrenheit: float) -> float:
        return (fahrenheit - 32) * 5/9
    
    @property
    def name(self) -> str:
        return "unit_converter"
    
    @property
    def description(self) -> str:
        return "Convert between cooking units (weight, volume, temperature). Supports cups, tablespoons, ounces, grams, celsius, fahrenheit, etc."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "value": {
                    "type": "number",
                    "description": "The value to convert"
                },
                "from_unit": {
                    "type": "string",
                    "description": "Source unit (e.g., 'cup', 'tbsp', 'g', 'celsius')"
                },
                "to_unit": {
                    "type": "string",
                    "description": "Target unit (e.g., 'ml', 'tsp', 'oz', 'fahrenheit')"
                },
                "ingredient": {
                    "type": "string",
                    "description": "Ingredient name for density-specific conversions",
                    "default": "water"
                }
            },
            "required": ["value", "from_unit", "to_unit"]
        }
    
    async def call(self, value: float, from_unit: str, to_unit: str, ingredient: str = "water") -> Dict[str, Any]:
        """Convert between cooking units."""
        try:
            from_unit = from_unit.lower().strip()
            to_unit = to_unit.lower().strip()
            
            # Temperature conversions
            if from_unit in ["celsius", "c"] and to_unit in ["fahrenheit", "f"]:
                result = self.celsius_to_fahrenheit(value)
                return {
                    "success": True,
                    "result": result,
                    "formatted": f"{value}°C = {result:.1f}°F"
                }
            elif from_unit in ["fahrenheit", "f"] and to_unit in ["celsius", "c"]:
                result = self.fahrenheit_to_celsius(value)
                return {
                    "success": True,
                    "result": result,
                    "formatted": f"{value}°F = {result:.1f}°C"
                }
            
            # Weight/Volume conversions
            if from_unit in self.UNIT_CONVERSIONS and to_unit in self.UNIT_CONVERSIONS:
                # Convert to base unit then to target
                base_value = value * self.UNIT_CONVERSIONS[from_unit]
                result = base_value / self.UNIT_CONVERSIONS[to_unit]
                
                return {
                    "success": True,
                    "result": result,
                    "formatted": f"{value} {from_unit} = {result:.2f} {to_unit}"
                }
            
            # Special conversions (ingredient-specific)
            if ingredient.lower() != "water":
                # Apply density corrections for common ingredients
                density_factor = self._get_ingredient_density_factor(ingredient)
                if density_factor != 1.0:
                    adjusted_result = result * density_factor
                    return {
                        "success": True,
                        "result": adjusted_result,
                        "formatted": f"{value} {from_unit} {ingredient} = {adjusted_result:.2f} {to_unit}",
                        "note": f"Adjusted for {ingredient} density"
                    }
            
            return {
                "success": False,
                "error": f"Cannot convert from {from_unit} to {to_unit}"
            }
            
        except Exception as e:
            logger.error(f"Unit conversion failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _get_ingredient_density_factor(self, ingredient: str) -> float:
        """Get density factor for ingredient-specific conversions."""
        ingredient_lower = ingredient.lower()
        
        # Common ingredient density factors (relative to water)
        density_factors = {
            "flour": 0.6,
            "sugar": 0.85,
            "brown sugar": 0.9,
            "butter": 0.95,
            "oil": 0.92,
            "honey": 1.4,
            "milk": 1.03,
            "cream": 1.01
        }
        
        for key, factor in density_factors.items():
            if key in ingredient_lower:
                return factor
        
        return 1.0  # Default to water density

class CookingTimerTool(BaseTool):
    """Tool for setting cooking timers and time calculations."""
    
    def __init__(self):
        self.active_timers = {}
    
    @property
    def name(self) -> str:
        return "cooking_timer"
    
    @property
    def description(self) -> str:
        return "Set cooking timers, calculate cooking times, and manage multiple timers for different cooking steps."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["set", "check", "list", "cancel", "calculate_time"],
                    "description": "Timer action to perform"
                },
                "duration_minutes": {
                    "type": "number",
                    "description": "Timer duration in minutes"
                },
                "timer_name": {
                    "type": "string",
                    "description": "Name for the timer (e.g., 'pasta', 'sauce')",
                    "default": "default"
                },
                "cooking_method": {
                    "type": "string",
                    "description": "Cooking method for time calculations",
                    "enum": ["boiling", "baking", "frying", "grilling", "steaming", "simmering"]
                },
                "ingredient": {
                    "type": "string",
                    "description": "Ingredient for cooking time estimation"
                },
                "quantity": {
                    "type": "string",
                    "description": "Quantity/size for cooking time estimation"
                }
            },
            "required": ["action"]
        }
    
    async def call(self, action: str, **kwargs) -> Dict[str, Any]:
        """Execute timer operations."""
        try:
            if action == "set":
                return await self._set_timer(kwargs)
            elif action == "check":
                return await self._check_timer(kwargs)
            elif action == "list":
                return await self._list_timers()
            elif action == "cancel":
                return await self._cancel_timer(kwargs)
            elif action == "calculate_time":
                return await self._calculate_cooking_time(kwargs)
            else:
                return {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            logger.error(f"Cooking timer operation failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _set_timer(self, params: Dict) -> Dict[str, Any]:
        """Set a cooking timer."""
        duration = params.get("duration_minutes", 0)
        timer_name = params.get("timer_name", "default")
        
        if duration <= 0:
            return {"success": False, "error": "Duration must be positive"}
        
        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration)
        
        self.active_timers[timer_name] = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_minutes": duration,
            "name": timer_name
        }
        
        return {
            "success": True,
            "message": f"Timer '{timer_name}' set for {duration} minutes",
            "end_time": end_time.isoformat(),
            "timer_id": timer_name
        }
    
    async def _check_timer(self, params: Dict) -> Dict[str, Any]:
        """Check timer status."""
        timer_name = params.get("timer_name", "default")
        
        if timer_name not in self.active_timers:
            return {"success": False, "error": f"Timer '{timer_name}' not found"}
        
        timer = self.active_timers[timer_name]
        now = datetime.now()
        
        if now >= timer["end_time"]:
            # Timer finished
            del self.active_timers[timer_name]
            return {
                "success": True,
                "status": "finished",
                "message": f"Timer '{timer_name}' has finished!",
                "overdue_seconds": (now - timer["end_time"]).total_seconds()
            }
        else:
            # Timer still running
            remaining = timer["end_time"] - now
            return {
                "success": True,
                "status": "running",
                "remaining_minutes": remaining.total_seconds() / 60,
                "end_time": timer["end_time"].isoformat()
            }
    
    async def _list_timers(self) -> Dict[str, Any]:
        """List all active timers."""
        now = datetime.now()
        timer_status = []
        
        for name, timer in self.active_timers.items():
            if now >= timer["end_time"]:
                status = "finished"
                remaining = 0
            else:
                status = "running"
                remaining = (timer["end_time"] - now).total_seconds() / 60
            
            timer_status.append({
                "name": name,
                "status": status,
                "remaining_minutes": remaining,
                "end_time": timer["end_time"].isoformat()
            })
        
        return {
            "success": True,
            "active_timers": timer_status,
            "count": len(timer_status)
        }
    
    async def _cancel_timer(self, params: Dict) -> Dict[str, Any]:
        """Cancel a timer."""
        timer_name = params.get("timer_name", "default")
        
        if timer_name not in self.active_timers:
            return {"success": False, "error": f"Timer '{timer_name}' not found"}
        
        del self.active_timers[timer_name]
        return {
            "success": True,
            "message": f"Timer '{timer_name}' cancelled"
        }
    
    async def _calculate_cooking_time(self, params: Dict) -> Dict[str, Any]:
        """Calculate estimated cooking time for ingredients."""
        cooking_method = params.get("cooking_method", "")
        ingredient = params.get("ingredient", "")
        quantity = params.get("quantity", "")
        
        # Basic cooking time estimates (in minutes)
        cooking_times = {
            "pasta": {"boiling": 8-12},
            "rice": {"boiling": 18, "steaming": 20},
            "chicken breast": {"baking": 25, "grilling": 15, "frying": 12},
            "salmon": {"baking": 15, "grilling": 10, "frying": 8},
            "vegetables": {"steaming": 8, "boiling": 5, "frying": 6},
            "potato": {"baking": 45, "boiling": 20, "frying": 15}
        }
        
        ingredient_lower = ingredient.lower()
        estimated_time = None
        
        for key, methods in cooking_times.items():
            if key in ingredient_lower:
                estimated_time = methods.get(cooking_method.lower())
                break
        
        if estimated_time:
            if isinstance(estimated_time, str) and "-" in estimated_time:
                # Handle ranges like "8-12"
                min_time, max_time = map(int, estimated_time.split("-"))
                estimated_time = (min_time + max_time) / 2
            
            return {
                "success": True,
                "estimated_minutes": estimated_time,
                "ingredient": ingredient,
                "cooking_method": cooking_method,
                "note": f"Estimated time for {cooking_method} {ingredient}: {estimated_time} minutes"
            }
        else:
            return {
                "success": False,
                "error": f"No cooking time data for {cooking_method} {ingredient}"
            }

class IngredientSubstitutionTool(BaseTool):
    """Tool for finding ingredient substitutions."""
    
    def __init__(self):
        self.substitution_engine = SubstitutionEngine()
    
    @property
    def name(self) -> str:
        return "ingredient_substitution"
    
    @property
    def description(self) -> str:
        return "Find ingredient substitutions based on dietary restrictions, availability, and cooking context."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ingredient": {
                    "type": "string",
                    "description": "Ingredient to substitute"
                },
                "dietary_restrictions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Dietary restrictions (vegan, gluten-free, etc.)"
                },
                "available_ingredients": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Available ingredients to use as substitutes"
                },
                "recipe_context": {
                    "type": "string",
                    "description": "Recipe context (baking, frying, salad, etc.)"
                }
            },
            "required": ["ingredient"]
        }
    
    async def call(self, ingredient: str, **kwargs) -> Dict[str, Any]:
        """Find ingredient substitutions."""
        try:
            # Create substitution request
            request = type('SubstitutionRequest', (), {
                'ingredient': ingredient,
                'dietary_restrictions': kwargs.get('dietary_restrictions', []),
                'available_ingredients': kwargs.get('available_ingredients', []),
                'recipe_context': kwargs.get('recipe_context', '')
            })()
            
            result = await self.substitution_engine.find_substitutions(request)
            
            return {
                "success": True,
                "substitutions": result
            }
            
        except Exception as e:
            logger.error(f"Ingredient substitution failed: {e}")
            return {"success": False, "error": str(e)}

class RecipeScalingTool(BaseTool):
    """Tool for scaling recipe quantities up or down."""
    
    @property
    def name(self) -> str:
        return "recipe_scaler"
    
    @property
    def description(self) -> str:
        return "Scale recipe ingredients up or down for different serving sizes."
    
    @property
    def parameters(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "amount": {"type": "number"},
                            "unit": {"type": "string"}
                        }
                    },
                    "description": "List of ingredients with amounts"
                },
                "original_servings": {
                    "type": "number",
                    "description": "Original number of servings"
                },
                "target_servings": {
                    "type": "number",
                    "description": "Desired number of servings"
                }
            },
            "required": ["ingredients", "original_servings", "target_servings"]
        }
    
    async def call(self, ingredients: List[Dict], original_servings: float, target_servings: float) -> Dict[str, Any]:
        """Scale recipe ingredients."""
        try:
            if original_servings <= 0 or target_servings <= 0:
                return {"success": False, "error": "Servings must be positive"}
            
            scale_factor = target_servings / original_servings
            scaled_ingredients = []
            
            for ingredient in ingredients:
                scaled_amount = ingredient["amount"] * scale_factor
                
                # Round to reasonable precision
                if scaled_amount < 0.1:
                    scaled_amount = round(scaled_amount, 3)
                elif scaled_amount < 1:
                    scaled_amount = round(scaled_amount, 2)
                else:
                    scaled_amount = round(scaled_amount, 1)
                
                scaled_ingredients.append({
                    "name": ingredient["name"],
                    "amount": scaled_amount,
                    "unit": ingredient["unit"],
                    "original_amount": ingredient["amount"]
                })
            
            return {
                "success": True,
                "scaled_ingredients": scaled_ingredients,
                "scale_factor": scale_factor,
                "original_servings": original_servings,
                "target_servings": target_servings
            }
            
        except Exception as e:
            logger.error(f"Recipe scaling failed: {e}")
            return {"success": False, "error": str(e)}

class ToolSystem:
    """Central tool management system."""
    
    def __init__(self):
        self.tools = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools."""
        default_tools = [
            NutritionCalculatorTool(),
            UnitConversionTool(),
            CookingTimerTool(),
            IngredientSubstitutionTool(),
            RecipeScalingTool()
        ]
        
        for tool in default_tools:
            self.register_tool(tool)
    
    def register_tool(self, tool: BaseTool):
        """Register a new tool."""
        self.tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name}")
    
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get OpenAI function calling schema for all tools."""
        schemas = []
        for tool in self.tools.values():
            schema = {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters
                }
            }
            schemas.append(schema)
        return schemas
    
    async def execute_tool_call(self, tool_call: ToolCall) -> ToolResult:
        """Execute a tool call and return the result."""
        try:
            if tool_call.name not in self.tools:
                return ToolResult(
                    call_id=tool_call.call_id,
                    name=tool_call.name,
                    result=None,
                    success=False,
                    error=f"Tool '{tool_call.name}' not found"
                )
            
            tool = self.tools[tool_call.name]
            result = await tool.call(**tool_call.arguments)
            
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                result=result,
                success=result.get("success", True) if isinstance(result, dict) else True
            )
            
        except Exception as e:
            logger.error(f"Tool execution failed: {e}")
            return ToolResult(
                call_id=tool_call.call_id,
                name=tool_call.name,
                result=None,
                success=False,
                error=str(e)
            )
    
    async def execute_multiple_tools(self, tool_calls: List[ToolCall]) -> List[ToolResult]:
        """Execute multiple tool calls concurrently."""
        tasks = [self.execute_tool_call(call) for call in tool_calls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                processed_results.append(ToolResult(
                    call_id=tool_calls[i].call_id,
                    name=tool_calls[i].name,
                    result=None,
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    def get_available_tools(self) -> Dict[str, str]:
        """Get list of available tools with descriptions."""
        return {name: tool.description for name, tool in self.tools.items()}