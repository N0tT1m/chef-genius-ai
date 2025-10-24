from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.database import get_db
from app.models.recipe import Recipe, RecipeCreate, RecipeResponse, RecipeGenerationRequest
from app.services.recipe_generator import RecipeGeneratorService
from app.services.nutrition_analyzer import NutritionAnalyzer

router = APIRouter()

@router.post("/generate", response_model=RecipeResponse)
async def generate_recipe(
    request: RecipeGenerationRequest,
    db: Session = Depends(get_db)
):
    """
    Generate a new recipe based on specified criteria.
    """
    try:
        generator = RecipeGeneratorService()
        recipe_data = await generator.generate_recipe(request)
        
        # Calculate nutrition
        nutrition_analyzer = NutritionAnalyzer()
        nutrition = nutrition_analyzer.analyze_recipe(recipe_data.ingredients)
        recipe_data.nutrition = nutrition
        
        # Save to database
        db_recipe = Recipe(**recipe_data.model_dump())
        db.add(db_recipe)
        db.commit()
        db.refresh(db_recipe)
        
        return RecipeResponse.model_validate(db_recipe)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recipe generation failed: {str(e)}")

@router.get("/", response_model=List[RecipeResponse])
async def get_recipes(
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    cuisine: Optional[str] = None,
    dietary_tags: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """
    Retrieve recipes with optional filtering.
    """
    query = db.query(Recipe)
    
    if cuisine:
        query = query.filter(Recipe.cuisine == cuisine)
    
    if dietary_tags:
        # Simple contains check - in production would use proper JSON querying
        query = query.filter(Recipe.dietary_tags.contains([dietary_tags]))
    
    recipes = query.offset(skip).limit(limit).all()
    return [RecipeResponse.model_validate(recipe) for recipe in recipes]

@router.get("/{recipe_id}", response_model=RecipeResponse)
async def get_recipe(recipe_id: int, db: Session = Depends(get_db)):
    """
    Get a specific recipe by ID.
    """
    recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return RecipeResponse.model_validate(recipe)

@router.post("/", response_model=RecipeResponse)
async def create_recipe(recipe: RecipeCreate, db: Session = Depends(get_db)):
    """
    Create a new recipe manually.
    """
    db_recipe = Recipe(**recipe.model_dump())
    db_recipe.is_generated = False
    db_recipe.source = "user-submitted"
    
    db.add(db_recipe)
    db.commit()
    db.refresh(db_recipe)
    
    return RecipeResponse.model_validate(db_recipe)

@router.put("/{recipe_id}", response_model=RecipeResponse)
async def update_recipe(
    recipe_id: int,
    recipe_update: RecipeCreate,
    db: Session = Depends(get_db)
):
    """
    Update an existing recipe.
    """
    db_recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not db_recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    for field, value in recipe_update.model_dump(exclude_unset=True).items():
        setattr(db_recipe, field, value)
    
    db.commit()
    db.refresh(db_recipe)
    
    return RecipeResponse.model_validate(db_recipe)

@router.delete("/{recipe_id}")
async def delete_recipe(recipe_id: int, db: Session = Depends(get_db)):
    """
    Delete a recipe.
    """
    db_recipe = db.query(Recipe).filter(Recipe.id == recipe_id).first()
    if not db_recipe:
        raise HTTPException(status_code=404, detail="Recipe not found")
    
    db.delete(db_recipe)
    db.commit()
    
    return {"message": "Recipe deleted successfully"}