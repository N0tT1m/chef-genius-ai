from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
import logging
from app.services.vision_service import VisionService
from app.services.multimodal_service import MultimodalFoodService

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize services
multimodal_service = MultimodalFoodService()

class IngredientRecognitionResponse(BaseModel):
    ingredients: List[Dict[str, Any]]
    confidence_scores: Dict[str, float]
    suggestions: List[str]

class RecipeReconstructionResponse(BaseModel):
    estimated_recipe: Dict[str, Any]
    confidence_score: float
    missing_information: List[str]

@router.post("/identify-ingredients", response_model=IngredientRecognitionResponse)
async def identify_ingredients(
    image: UploadFile = File(...),
    context: Optional[str] = Form(None)
):
    """
    Identify ingredients from an uploaded image.
    """
    try:
        # Validate file type
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Process with vision service
        vision_service = VisionService()
        result = await vision_service.identify_ingredients(image_data, context)
        
        return result
        
    except Exception as e:
        logger.error(f"Ingredient identification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@router.post("/scan-fridge")
async def scan_fridge(
    image: UploadFile = File(...),
    generate_recipes: bool = Form(True)
):
    """
    Scan fridge contents and optionally generate recipe suggestions.
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await image.read()
        
        vision_service = VisionService()
        fridge_contents = await vision_service.scan_fridge(image_data)
        
        result = {
            "detected_ingredients": fridge_contents["ingredients"],
            "freshness_assessment": fridge_contents.get("freshness", {}),
            "organization_tips": fridge_contents.get("tips", [])
        }
        
        if generate_recipes:
            # Generate recipe suggestions based on detected ingredients
            recipes = await vision_service.suggest_recipes_from_ingredients(
                fridge_contents["ingredients"]
            )
            result["recipe_suggestions"] = recipes
        
        return result
        
    except Exception as e:
        logger.error(f"Fridge scanning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fridge scan failed: {str(e)}")

@router.post("/analyze-cooking-progress")
async def analyze_cooking_progress(
    image: UploadFile = File(...),
    recipe_step: Optional[str] = Form(None),
    expected_result: Optional[str] = Form(None)
):
    """
    Analyze cooking progress and provide feedback.
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await image.read()
        
        vision_service = VisionService()
        analysis = await vision_service.analyze_cooking_progress(
            image_data, recipe_step, expected_result
        )
        
        return analysis
        
    except Exception as e:
        logger.error(f"Cooking progress analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/analyze-plating")
async def analyze_plating(
    image: UploadFile = File(...),
    dish_type: Optional[str] = Form(None)
):
    """
    Analyze food plating and provide presentation suggestions.
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await image.read()
        
        vision_service = VisionService()
        analysis = await vision_service.analyze_plating(image_data, dish_type)
        
        return analysis
        
    except Exception as e:
        logger.error(f"Plating analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/reconstruct-recipe", response_model=RecipeReconstructionResponse)
async def reconstruct_recipe(
    image: UploadFile = File(...),
    additional_info: Optional[str] = Form(None)
):
    """
    Attempt to reconstruct a recipe from an image of the finished dish.
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await image.read()
        
        vision_service = VisionService()
        reconstruction = await vision_service.reconstruct_recipe(image_data, additional_info)
        
        return reconstruction
        
    except Exception as e:
        logger.error(f"Recipe reconstruction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Reconstruction failed: {str(e)}")

@router.post("/identify-cooking-technique")
async def identify_cooking_technique(
    image: UploadFile = File(...),
    context: Optional[str] = Form(None)
):
    """
    Identify cooking techniques being used in an image.
    """
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        image_data = await image.read()
        
        vision_service = VisionService()
        techniques = await vision_service.identify_cooking_technique(image_data, context)
        
        return {
            "detected_techniques": techniques["techniques"],
            "confidence_scores": techniques["confidence"],
            "recommendations": techniques.get("recommendations", []),
            "next_steps": techniques.get("next_steps", [])
        }
        
    except Exception as e:
        logger.error(f"Technique identification failed: {e}")
        raise HTTPException(status_code=500, detail=f"Technique identification failed: {str(e)}")

# New Advanced Multimodal Endpoints

@router.post("/analyze-food-multimodal")
async def analyze_food_multimodal(
    image: UploadFile = File(...),
    analysis_type: str = Query(default="comprehensive", description="comprehensive, ingredients, cooking_stage, dish_type, caption, objects, quality, nutrition")
):
    """Advanced multimodal food analysis using modern AI models."""
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Perform multimodal analysis
        result = await multimodal_service.analyze_food_image(image_data, analysis_type)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "analysis_type": analysis_type,
            "results": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multimodal food analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@router.post("/compare-dishes")
async def compare_dishes(
    image1: UploadFile = File(..., description="First food image"),
    image2: UploadFile = File(..., description="Second food image")
):
    """Compare two food images for similarity."""
    try:
        if not image1.content_type.startswith("image/") or not image2.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Both files must be images")
        
        # Read image data
        image1_data = await image1.read()
        image2_data = await image2.read()
        
        # Compare images
        result = await multimodal_service.compare_images(image1_data, image2_data)
        
        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])
        
        return {
            "status": "success",
            "comparison": result
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image comparison failed: {e}")
        raise HTTPException(status_code=500, detail=f"Comparison failed: {str(e)}")

@router.post("/generate-recipe-from-image-ai")
async def generate_recipe_from_image_ai(
    image: UploadFile = File(...),
    dietary_restrictions: Optional[List[str]] = Query(default=None),
    cuisine_preference: Optional[str] = Query(default=None),
    difficulty: Optional[str] = Query(default="medium")
):
    """Generate a recipe based on food image analysis using AI."""
    try:
        if not image.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await image.read()
        
        # Analyze image comprehensively
        analysis = await multimodal_service.analyze_food_image(image_data, "comprehensive")
        
        if "error" in analysis:
            raise HTTPException(status_code=500, detail=f"Image analysis failed: {analysis['error']}")
        
        # Extract ingredients and context for recipe generation
        detected_ingredients = []
        if "detected_ingredients" in analysis:
            detected_ingredients = [ing["ingredient"] for ing in analysis["detected_ingredients"][:5]]
        
        # Determine cuisine from image if not specified
        recipe_cuisine = cuisine_preference
        if not recipe_cuisine and "dish_classification" in analysis:
            dish_type = analysis["dish_classification"].get("primary_dish_type", "")
            if "pasta" in dish_type:
                recipe_cuisine = "Italian"
            elif "curry" in dish_type:
                recipe_cuisine = "Indian"
            elif "stir fry" in dish_type:
                recipe_cuisine = "Asian"
        
        return {
            "status": "success",
            "image_analysis": {
                "detected_ingredients": analysis.get("detected_ingredients", []),
                "cooking_stage": analysis.get("cooking_stage", {}),
                "dish_type": analysis.get("dish_classification", {}),
                "caption": analysis.get("caption", ""),
                "recipe_suggestions": analysis.get("recipe_suggestions", [])
            },
            "recipe_parameters": {
                "ingredients": detected_ingredients,
                "cuisine": recipe_cuisine,
                "dietary_restrictions": dietary_restrictions or [],
                "difficulty": difficulty
            },
            "message": "Use the /recipes/generate endpoint with these parameters to generate the full recipe"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI recipe generation from image failed: {e}")
        raise HTTPException(status_code=500, detail=f"Recipe generation failed: {str(e)}")

@router.get("/multimodal-models-status")
async def get_multimodal_models_status():
    """Get the status of all multimodal AI models."""
    try:
        status = multimodal_service.get_model_status()
        return {
            "status": "success",
            "models": status
        }
        
    except Exception as e:
        logger.error(f"Failed to get model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))