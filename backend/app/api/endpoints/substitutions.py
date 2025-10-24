from fastapi import APIRouter, HTTPException, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.substitution_engine import SubstitutionEngine

router = APIRouter()

class SubstitutionRequest(BaseModel):
    ingredient: str
    quantity: Optional[str] = None
    recipe_context: Optional[str] = None
    dietary_restrictions: Optional[List[str]] = []
    available_ingredients: Optional[List[str]] = []

class SubstitutionResponse(BaseModel):
    original_ingredient: str
    substitutes: List[Dict[str, Any]]
    notes: List[str]

@router.post("/find", response_model=SubstitutionResponse)
async def find_substitutions(request: SubstitutionRequest):
    """
    Find suitable substitutions for an ingredient.
    """
    try:
        engine = SubstitutionEngine()
        result = await engine.find_substitutions(request)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Substitution search failed: {str(e)}")

@router.get("/compatibility")
async def check_compatibility(
    ingredient1: str = Query(..., description="First ingredient"),
    ingredient2: str = Query(..., description="Second ingredient")
):
    """
    Check flavor compatibility between two ingredients.
    """
    try:
        engine = SubstitutionEngine()
        compatibility = engine.check_flavor_compatibility(ingredient1, ingredient2)
        return {
            "ingredient1": ingredient1,
            "ingredient2": ingredient2,
            "compatibility_score": compatibility["score"],
            "explanation": compatibility["explanation"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compatibility check failed: {str(e)}")

@router.post("/adapt-recipe")
async def adapt_recipe_for_diet(
    recipe_id: Optional[int] = None,
    ingredients: Optional[List[Dict[str, Any]]] = None,
    target_diet: str = Query(..., description="Target dietary restriction"),
):
    """
    Adapt a recipe to meet specific dietary requirements.
    """
    try:
        engine = SubstitutionEngine()
        if recipe_id:
            adapted_recipe = await engine.adapt_recipe_by_id(recipe_id, target_diet)
        elif ingredients:
            adapted_recipe = await engine.adapt_ingredients(ingredients, target_diet)
        else:
            raise HTTPException(status_code=400, detail="Either recipe_id or ingredients must be provided")
        
        return adapted_recipe
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recipe adaptation failed: {str(e)}")

@router.get("/seasonal-alternatives")
async def get_seasonal_alternatives(
    ingredient: str = Query(..., description="Ingredient to find alternatives for"),
    season: Optional[str] = Query(None, description="Target season (spring, summer, fall, winter)")
):
    """
    Get seasonal alternatives for an ingredient.
    """
    try:
        engine = SubstitutionEngine()
        alternatives = engine.get_seasonal_alternatives(ingredient, season)
        return {
            "ingredient": ingredient,
            "season": season or "current",
            "alternatives": alternatives
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Seasonal alternatives search failed: {str(e)}")