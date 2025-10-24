from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from pydantic import BaseModel
from app.core.database import get_db
from app.services.meal_planner import MealPlannerService

router = APIRouter()

class MealPlanRequest(BaseModel):
    days: int = 7
    meals_per_day: List[str] = ["breakfast", "lunch", "dinner"]
    people: int = 2
    dietary_restrictions: Optional[List[str]] = []
    cuisine_preferences: Optional[List[str]] = []
    target_calories: Optional[int] = None
    prep_time_limit: Optional[int] = None  # in minutes
    budget_range: Optional[str] = None  # "low", "medium", "high"
    cooking_skill: Optional[str] = "medium"  # "beginner", "medium", "advanced"
    exclude_ingredients: Optional[List[str]] = []

class MealPlanResponse(BaseModel):
    id: Optional[int] = None
    plan_name: str
    start_date: datetime
    end_date: datetime
    total_days: int
    meals: List[Dict[str, Any]]
    shopping_list: Dict[str, List[Dict[str, Any]]]
    nutrition_summary: Dict[str, Any]
    prep_schedule: List[Dict[str, Any]]
    estimated_cost: Optional[float] = None

@router.post("/generate", response_model=MealPlanResponse)
async def generate_meal_plan(
    request: MealPlanRequest,
    db: Session = Depends(get_db)
):
    """
    Generate a personalized meal plan.
    """
    try:
        planner = MealPlannerService()
        meal_plan = await planner.generate_meal_plan(request)
        
        # Save to database (implementation would store in meal_plans table)
        
        return meal_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meal plan generation failed: {str(e)}")

@router.get("/", response_model=List[MealPlanResponse])
async def get_meal_plans(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=50),
    db: Session = Depends(get_db)
):
    """
    Get existing meal plans.
    """
    # Implementation would query meal_plans table
    return []

@router.get("/{plan_id}", response_model=MealPlanResponse)
async def get_meal_plan(plan_id: int, db: Session = Depends(get_db)):
    """
    Get a specific meal plan by ID.
    """
    # Implementation would query specific meal plan
    raise HTTPException(status_code=404, detail="Meal plan not found")

@router.post("/{plan_id}/shopping-list")
async def generate_shopping_list(
    plan_id: int,
    consolidate_by_store: bool = Query(False),
    db: Session = Depends(get_db)
):
    """
    Generate or update shopping list for a meal plan.
    """
    try:
        planner = MealPlannerService()
        shopping_list = await planner.generate_shopping_list(plan_id, consolidate_by_store)
        return shopping_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Shopping list generation failed: {str(e)}")

@router.put("/{plan_id}/meals/{meal_id}/substitute")
async def substitute_meal(
    plan_id: int,
    meal_id: int,
    dietary_restrictions: Optional[List[str]] = None,
    cuisine_preference: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Substitute a specific meal in the plan with an alternative.
    """
    try:
        planner = MealPlannerService()
        updated_plan = await planner.substitute_meal(
            plan_id, meal_id, dietary_restrictions, cuisine_preference
        )
        return updated_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Meal substitution failed: {str(e)}")

@router.get("/nutrition/analysis")
async def analyze_plan_nutrition(
    plan_id: int = Query(...),
    target_calories: Optional[int] = None,
    target_protein: Optional[int] = None,
    db: Session = Depends(get_db)
):
    """
    Analyze the nutritional content of a meal plan.
    """
    try:
        planner = MealPlannerService()
        analysis = await planner.analyze_nutrition(plan_id, target_calories, target_protein)
        return analysis
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Nutrition analysis failed: {str(e)}")

@router.post("/optimize")
async def optimize_meal_plan(
    plan_id: int,
    optimization_goals: List[str] = Query(...),  # "cost", "nutrition", "time", "variety"
    constraints: Optional[Dict[str, Any]] = None,
    db: Session = Depends(get_db)
):
    """
    Optimize an existing meal plan based on specified goals.
    """
    try:
        planner = MealPlannerService()
        optimized_plan = await planner.optimize_plan(plan_id, optimization_goals, constraints)
        return optimized_plan
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Plan optimization failed: {str(e)}")