"""
Rust Core Integration Service

This module provides a seamless integration between the Python backend
and the high-performance Rust core library.
"""

import logging
from typing import List, Dict, Any, Optional, Union
from contextlib import asynccontextmanager
import asyncio
from functools import wraps
import time

from app.models.recipe import RecipeGenerationRequest, RecipeCreate, NutritionInfo, Ingredient
from app.core.config import settings

logger = logging.getLogger(__name__)

# Try to import the Rust core - fall back gracefully if not available
try:
    import chef_genius_core as cgc
    RUST_CORE_AVAILABLE = True
    logger.info("✅ Rust core library loaded successfully")
except ImportError as e:
    RUST_CORE_AVAILABLE = False
    logger.warning(f"⚠️  Rust core library not available: {e}")
    logger.warning("Falling back to Python implementations")


def require_rust_core(func):
    """Decorator to ensure Rust core is available"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not RUST_CORE_AVAILABLE:
            raise RuntimeError("Rust core library is not available. Install with: pip install chef-genius-core")
        return func(*args, **kwargs)
    return wrapper


class RustCoreManager:
    """Manager for Rust core engines with fallback support"""
    
    def __init__(self):
        self.inference_engine = None
        self.search_engine = None
        self.recipe_processor = None
        self.nutrition_analyzer = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize all Rust engines"""
        if not RUST_CORE_AVAILABLE:
            logger.warning("Rust core not available - using Python fallbacks")
            return
            
        try:
            # Initialize engines
            self.inference_engine = cgc.PyInferenceEngine()
            self.search_engine = cgc.PyVectorSearchEngine()
            self.recipe_processor = cgc.PyRecipeProcessor()
            self.nutrition_analyzer = cgc.PyNutritionAnalyzer()
            
            # Warm up the inference engine
            await self._warmup_inference_engine()
            
            self.initialized = True
            logger.info("🚀 Rust core engines initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize Rust core: {e}")
            self.initialized = False
            
    async def _warmup_inference_engine(self):
        """Warm up the inference engine with sample data"""
        if self.inference_engine:
            try:
                sample_request = cgc.PyInferenceRequest(
                    ingredients=["chicken", "rice", "vegetables"],
                    max_length=256,
                    temperature=0.8,
                    use_cache=False
                )
                self.inference_engine.warmup(sample_request.ingredients)
                logger.info("Inference engine warmed up successfully")
            except Exception as e:
                logger.warning(f"Failed to warm up inference engine: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        if not self.initialized:
            return {"status": "not_initialized"}
            
        stats = {
            "status": "initialized",
            "rust_core_available": RUST_CORE_AVAILABLE,
            "system_info": cgc.get_system_info() if RUST_CORE_AVAILABLE else {},
        }
        
        if self.inference_engine:
            stats["inference"] = self.inference_engine.get_stats()
            
        if self.search_engine:
            stats["search"] = self.search_engine.get_stats()
            
        if self.recipe_processor:
            stats["recipe_processing"] = self.recipe_processor.get_stats()
            
        if self.nutrition_analyzer:
            stats["nutrition"] = self.nutrition_analyzer.get_stats()
            
        return stats


# Global manager instance
rust_manager = RustCoreManager()


class RustInferenceService:
    """High-performance recipe generation using Rust core"""
    
    def __init__(self):
        self.fallback_service = None  # Would import Python fallback service
        
    @require_rust_core
    async def generate_recipe(
        self, 
        request: RecipeGenerationRequest,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """Generate recipe using Rust inference engine"""
        start_time = time.time()
        
        try:
            # Convert Python request to Rust request
            rust_request = cgc.PyInferenceRequest(
                ingredients=request.ingredients,
                max_length=getattr(request, 'max_length', 512),
                temperature=getattr(request, 'temperature', 0.8),
                top_k=getattr(request, 'top_k', 50),
                top_p=getattr(request, 'top_p', 0.9),
                cuisine_style=getattr(request, 'cuisine_style', None),
                dietary_restrictions=getattr(request, 'dietary_restrictions', None),
                use_cache=use_cache
            )
            
            # Generate using Rust engine
            response = rust_manager.inference_engine.generate_recipe(rust_request)
            
            # Convert Rust response to Python format
            result = {
                "recipe": {
                    "title": response.recipe.title,
                    "ingredients": response.recipe.ingredients,
                    "instructions": response.recipe.instructions,
                    "cooking_time": response.recipe.cooking_time,
                    "prep_time": response.recipe.prep_time,
                    "servings": response.recipe.servings,
                    "difficulty": response.recipe.difficulty,
                    "cuisine_type": response.recipe.cuisine_type,
                    "dietary_tags": response.recipe.dietary_tags,
                    "confidence": response.recipe.confidence,
                },
                "confidence": response.confidence,
                "generation_time_ms": response.generation_time_ms,
                "model_version": response.model_version,
                "cached": response.cached,
                "performance": {
                    "total_time_ms": (time.time() - start_time) * 1000,
                    "rust_time_ms": response.generation_time_ms,
                    "python_overhead_ms": (time.time() - start_time) * 1000 - response.generation_time_ms
                }
            }
            
            logger.info(f"Recipe generated in {result['performance']['total_time_ms']:.1f}ms (Rust: {response.generation_time_ms}ms)")
            return result
            
        except Exception as e:
            logger.error(f"Rust inference failed: {e}")
            raise


class RustSearchService:
    """High-performance vector search using Rust core"""
    
    def __init__(self):
        self.initialized = False
        
    async def initialize_index(self, recipes: List[Dict[str, Any]]):
        """Initialize search index with recipes"""
        if not rust_manager.initialized:
            return
            
        try:
            # Convert Python recipes to Rust format
            rust_recipes = []
            for recipe in recipes:
                rust_recipe = cgc.PyRecipe(
                    title=recipe.get("title", ""),
                    ingredients=recipe.get("ingredients", []),
                    instructions=recipe.get("instructions", []),
                    cooking_time=recipe.get("cooking_time"),
                    prep_time=recipe.get("prep_time"),
                    servings=recipe.get("servings"),
                    difficulty=recipe.get("difficulty"),
                    cuisine_type=recipe.get("cuisine_type"),
                    dietary_tags=recipe.get("dietary_tags"),
                    confidence=recipe.get("confidence")
                )
                rust_recipes.append(rust_recipe)
            
            # Add to search index
            rust_manager.search_engine.add_recipes(rust_recipes)
            self.initialized = True
            logger.info(f"Search index initialized with {len(recipes)} recipes")
            
        except Exception as e:
            logger.error(f"Failed to initialize search index: {e}")
    
    @require_rust_core
    async def search_recipes(
        self, 
        query: str, 
        top_k: int = 10,
        filters: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """Search for recipes using semantic similarity"""
        try:
            results = rust_manager.search_engine.search(query, top_k, filters)
            
            # Convert Rust results to Python format
            python_results = []
            for result in results:
                python_result = {
                    "recipe": {
                        "title": result.recipe.title,
                        "ingredients": result.recipe.ingredients,
                        "instructions": result.recipe.instructions,
                        "cooking_time": result.recipe.cooking_time,
                        "servings": result.recipe.servings,
                        "difficulty": result.recipe.difficulty,
                        "cuisine_type": result.recipe.cuisine_type,
                    },
                    "score": result.score,
                    "distance": result.distance,
                    "metadata": result.metadata
                }
                python_results.append(python_result)
                
            return python_results
            
        except Exception as e:
            logger.error(f"Rust search failed: {e}")
            raise
    
    @require_rust_core
    async def search_by_ingredients(
        self, 
        ingredients: List[str], 
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Search for recipes by ingredients"""
        try:
            results = rust_manager.search_engine.search_by_ingredients(ingredients, top_k)
            
            # Convert to Python format
            python_results = []
            for result in results:
                python_result = {
                    "recipe": {
                        "title": result.recipe.title,
                        "ingredients": result.recipe.ingredients,
                        "instructions": result.recipe.instructions,
                        "cooking_time": result.recipe.cooking_time,
                        "servings": result.recipe.servings,
                        "difficulty": result.recipe.difficulty,
                        "cuisine_type": result.recipe.cuisine_type,
                    },
                    "score": result.score,
                    "distance": result.distance,
                }
                python_results.append(python_result)
                
            return python_results
            
        except Exception as e:
            logger.error(f"Ingredient search failed: {e}")
            raise


class RustRecipeProcessor:
    """High-performance recipe processing using Rust core"""
    
    @require_rust_core
    async def parse_ingredients(self, text: str) -> List[Dict[str, Any]]:
        """Parse ingredient text into structured data"""
        try:
            ingredients = rust_manager.recipe_processor.parse_ingredients(text)
            return ingredients
        except Exception as e:
            logger.error(f"Ingredient parsing failed: {e}")
            raise
    
    @require_rust_core
    async def find_substitutions(
        self, 
        ingredient: str, 
        dietary_restrictions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Find ingredient substitutions"""
        try:
            substitutions = rust_manager.recipe_processor.find_substitutions(
                ingredient, dietary_restrictions
            )
            return substitutions
        except Exception as e:
            logger.error(f"Substitution search failed: {e}")
            raise
    
    @require_rust_core
    async def process_recipe_text(self, text: str) -> Dict[str, Any]:
        """Process and clean recipe text"""
        try:
            cleaned = rust_manager.recipe_processor.clean_recipe_text(text)
            steps = rust_manager.recipe_processor.extract_cooking_steps(cleaned)
            time_estimate = rust_manager.recipe_processor.estimate_cooking_time(steps)
            
            return {
                "cleaned_text": cleaned,
                "cooking_steps": steps,
                "time_estimate": time_estimate
            }
        except Exception as e:
            logger.error(f"Recipe processing failed: {e}")
            raise


class RustNutritionService:
    """High-performance nutrition analysis using Rust core"""
    
    @require_rust_core
    async def analyze_recipe(self, recipe: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze nutrition for a recipe"""
        try:
            # Convert to Rust recipe format
            rust_recipe = cgc.PyRecipe(
                title=recipe.get("title", ""),
                ingredients=recipe.get("ingredients", []),
                instructions=recipe.get("instructions", []),
                cooking_time=recipe.get("cooking_time"),
                prep_time=recipe.get("prep_time"),
                servings=recipe.get("servings"),
                difficulty=recipe.get("difficulty"),
                cuisine_type=recipe.get("cuisine_type"),
                dietary_tags=recipe.get("dietary_tags"),
                confidence=recipe.get("confidence")
            )
            
            nutrition = rust_manager.nutrition_analyzer.analyze_recipe(rust_recipe)
            return nutrition.to_dict()
            
        except Exception as e:
            logger.error(f"Nutrition analysis failed: {e}")
            raise
    
    @require_rust_core
    async def get_allergens(self, ingredients: List[str]) -> List[str]:
        """Get allergen information for ingredients"""
        try:
            allergens = rust_manager.nutrition_analyzer.get_allergens(ingredients)
            return allergens
        except Exception as e:
            logger.error(f"Allergen analysis failed: {e}")
            raise


# Service instances
rust_inference_service = RustInferenceService()
rust_search_service = RustSearchService()
rust_recipe_processor = RustRecipeProcessor()
rust_nutrition_service = RustNutritionService()


@asynccontextmanager
async def rust_core_lifespan():
    """Context manager for Rust core lifecycle"""
    try:
        await rust_manager.initialize()
        yield rust_manager
    finally:
        # Cleanup if needed
        pass


async def initialize_rust_services():
    """Initialize all Rust services"""
    await rust_manager.initialize()
    logger.info("🚀 All Rust services initialized")


def is_rust_available() -> bool:
    """Check if Rust core is available and initialized"""
    return RUST_CORE_AVAILABLE and rust_manager.initialized


def get_rust_performance_stats() -> Dict[str, Any]:
    """Get comprehensive performance statistics"""
    return rust_manager.get_performance_stats()