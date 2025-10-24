"""
Tools API endpoints for Chef Genius

Provides API access to the tool calling system for nutrition calculations,
unit conversions, timers, and other cooking utilities.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import logging
from app.services.tool_system import ToolSystem, ToolCall, ToolResult
from app.core.dependencies import get_current_user

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize tool system
tool_system = ToolSystem()

class ToolCallRequest(BaseModel):
    """Request model for tool calls."""
    tool_name: str
    arguments: Dict[str, Any]

class MultipleToolCallRequest(BaseModel):
    """Request model for multiple tool calls."""
    tool_calls: List[ToolCallRequest]

class ToolCallResponse(BaseModel):
    """Response model for tool call results."""
    success: bool
    result: Any
    tool_name: str
    error: Optional[str] = None

@router.get("/available")
async def get_available_tools(current_user = Depends(get_current_user)):
    """Get list of all available tools with descriptions."""
    try:
        tools = tool_system.get_available_tools()
        return {
            "status": "success",
            "tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        logger.error(f"Failed to get available tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/schemas")
async def get_tool_schemas(current_user = Depends(get_current_user)):
    """Get OpenAI function calling schemas for all tools."""
    try:
        schemas = tool_system.get_tool_schemas()
        return {
            "status": "success",
            "schemas": schemas,
            "count": len(schemas)
        }
    except Exception as e:
        logger.error(f"Failed to get tool schemas: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/call", response_model=ToolCallResponse)
async def call_tool(
    request: ToolCallRequest,
    current_user = Depends(get_current_user)
):
    """Execute a single tool call."""
    try:
        tool_call = ToolCall(
            name=request.tool_name,
            arguments=request.arguments,
            call_id="single_call"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        return ToolCallResponse(
            success=result.success,
            result=result.result,
            tool_name=result.name,
            error=result.error
        )
        
    except Exception as e:
        logger.error(f"Tool call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/call-multiple")
async def call_multiple_tools(
    request: MultipleToolCallRequest,
    current_user = Depends(get_current_user)
):
    """Execute multiple tool calls concurrently."""
    try:
        tool_calls = [
            ToolCall(
                name=call.tool_name,
                arguments=call.arguments,
                call_id=f"call_{i}"
            )
            for i, call in enumerate(request.tool_calls)
        ]
        
        results = await tool_system.execute_multiple_tools(tool_calls)
        
        response_results = [
            ToolCallResponse(
                success=result.success,
                result=result.result,
                tool_name=result.name,
                error=result.error
            )
            for result in results
        ]
        
        return {
            "status": "success",
            "results": response_results,
            "count": len(response_results)
        }
        
    except Exception as e:
        logger.error(f"Multiple tool calls failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Specific tool endpoints for common operations

@router.post("/nutrition/calculate")
async def calculate_nutrition(
    ingredients: List[Dict[str, Any]],
    servings: int = Query(default=1, description="Number of servings"),
    current_user = Depends(get_current_user)
):
    """Calculate nutritional information for ingredients."""
    try:
        tool_call = ToolCall(
            name="nutrition_calculator",
            arguments={"ingredients": ingredients, "servings": servings},
            call_id="nutrition_calc"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        if result.success:
            return {
                "status": "success",
                "nutrition": result.result
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Nutrition calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/units/convert")
async def convert_units(
    value: float,
    from_unit: str,
    to_unit: str,
    ingredient: str = Query(default="water", description="Ingredient for density-specific conversions"),
    current_user = Depends(get_current_user)
):
    """Convert between cooking units."""
    try:
        tool_call = ToolCall(
            name="unit_converter",
            arguments={
                "value": value,
                "from_unit": from_unit,
                "to_unit": to_unit,
                "ingredient": ingredient
            },
            call_id="unit_convert"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        if result.success:
            return {
                "status": "success",
                "conversion": result.result
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Unit conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/timer/{action}")
async def manage_timer(
    action: str,
    duration_minutes: Optional[float] = Query(default=None, description="Timer duration in minutes"),
    timer_name: str = Query(default="default", description="Timer name"),
    current_user = Depends(get_current_user)
):
    """Manage cooking timers (set, check, list, cancel)."""
    try:
        arguments = {
            "action": action,
            "timer_name": timer_name
        }
        
        if duration_minutes is not None:
            arguments["duration_minutes"] = duration_minutes
        
        tool_call = ToolCall(
            name="cooking_timer",
            arguments=arguments,
            call_id="timer_action"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        if result.success:
            return {
                "status": "success",
                "timer": result.result
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Timer management failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/substitutions/find")
async def find_substitutions(
    ingredient: str,
    dietary_restrictions: Optional[List[str]] = Query(default=None),
    available_ingredients: Optional[List[str]] = Query(default=None),
    recipe_context: Optional[str] = Query(default=None),
    current_user = Depends(get_current_user)
):
    """Find ingredient substitutions."""
    try:
        arguments = {"ingredient": ingredient}
        
        if dietary_restrictions:
            arguments["dietary_restrictions"] = dietary_restrictions
        if available_ingredients:
            arguments["available_ingredients"] = available_ingredients
        if recipe_context:
            arguments["recipe_context"] = recipe_context
        
        tool_call = ToolCall(
            name="ingredient_substitution",
            arguments=arguments,
            call_id="substitution_find"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        if result.success:
            return {
                "status": "success",
                "substitutions": result.result
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Substitution search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recipe/scale")
async def scale_recipe(
    ingredients: List[Dict[str, Any]],
    original_servings: float,
    target_servings: float,
    current_user = Depends(get_current_user)
):
    """Scale recipe ingredients for different serving sizes."""
    try:
        tool_call = ToolCall(
            name="recipe_scaler",
            arguments={
                "ingredients": ingredients,
                "original_servings": original_servings,
                "target_servings": target_servings
            },
            call_id="recipe_scale"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        if result.success:
            return {
                "status": "success",
                "scaled_recipe": result.result
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Recipe scaling failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/cooking-time/calculate")
async def calculate_cooking_time(
    cooking_method: str,
    ingredient: str,
    quantity: Optional[str] = Query(default=None, description="Quantity/size description"),
    current_user = Depends(get_current_user)
):
    """Calculate estimated cooking time for ingredients."""
    try:
        arguments = {
            "action": "calculate_time",
            "cooking_method": cooking_method,
            "ingredient": ingredient
        }
        
        if quantity:
            arguments["quantity"] = quantity
        
        tool_call = ToolCall(
            name="cooking_timer",
            arguments=arguments,
            call_id="cooking_time_calc"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        if result.success:
            return {
                "status": "success",
                "cooking_time": result.result
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Cooking time calculation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Utility endpoints

@router.get("/temperature/convert")
async def convert_temperature(
    temperature: float,
    from_unit: str = Query(description="Source unit: celsius or fahrenheit"),
    to_unit: str = Query(description="Target unit: celsius or fahrenheit"),
    current_user = Depends(get_current_user)
):
    """Convert between Celsius and Fahrenheit."""
    try:
        tool_call = ToolCall(
            name="unit_converter",
            arguments={
                "value": temperature,
                "from_unit": from_unit,
                "to_unit": to_unit
            },
            call_id="temp_convert"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        if result.success:
            return {
                "status": "success",
                "temperature_conversion": result.result
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Temperature conversion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/timers/active")
async def get_active_timers(current_user = Depends(get_current_user)):
    """Get all active cooking timers."""
    try:
        tool_call = ToolCall(
            name="cooking_timer",
            arguments={"action": "list"},
            call_id="list_timers"
        )
        
        result = await tool_system.execute_tool_call(tool_call)
        
        if result.success:
            return {
                "status": "success",
                "timers": result.result
            }
        else:
            raise HTTPException(status_code=400, detail=result.error)
            
    except Exception as e:
        logger.error(f"Failed to get active timers: {e}")
        raise HTTPException(status_code=500, detail=str(e))