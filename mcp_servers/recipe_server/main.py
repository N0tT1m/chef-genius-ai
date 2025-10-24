"""
Recipe Generation MCP Server
Handles recipe generation using T5-Large model with RAG enhancement
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.types import Tool, TextContent
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from backend.app.services.enhanced_rag_system import EnhancedRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RecipeGenerationServer:
    """MCP Server for recipe generation with T5-Large and RAG enhancement."""
    
    def __init__(self, 
                 model_path: str = None,
                 rag_system: EnhancedRAGSystem = None,
                 device: str = "auto"):
        """
        Initialize the recipe generation server.
        
        Args:
            model_path: Path to the fine-tuned T5 model
            rag_system: Enhanced RAG system instance
            device: Device to run the model on
        """
        self.server = Server("recipe-generation")
        self.model_path = model_path or "/Users/timmy/workspace/ai-apps/chef-genius/models/recipe_generation"
        self.device = self._setup_device(device)
        
        # Model components
        self.model = None
        self.tokenizer = None
        self.rag_system = rag_system or EnhancedRAGSystem()
        
        # Generation settings
        self.max_length = 1024
        self.temperature = 0.7
        self.top_p = 0.9
        self.repetition_penalty = 1.2
        
        # Performance tracking
        self.generation_count = 0
        self.avg_generation_time = 0
        
        self._setup_tools()
        self._load_model()
    
    def _setup_device(self, device: str) -> str:
        """Setup the appropriate device for model inference."""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                device = "mps"
                logger.info("Using MPS device")
            else:
                device = "cpu"
                logger.info("Using CPU device")
        
        return device
    
    def _load_model(self):
        """Load the fine-tuned T5 model and tokenizer."""
        try:
            logger.info(f"Loading T5 model from {self.model_path}")
            
            # Load tokenizer
            self.tokenizer = T5Tokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = T5ForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Move to device
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"T5 model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load T5 model: {e}")
            # Fallback to base model
            self._load_base_model()
    
    def _load_base_model(self):
        """Load base T5-Large model as fallback."""
        try:
            logger.info("Loading base T5-Large model as fallback")
            
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.model = T5ForConditionalGeneration.from_pretrained(
                "t5-large",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Base T5-Large model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load base model: {e}")
            raise
    
    def _setup_tools(self):
        """Setup MCP tools for recipe generation."""
        
        @self.server.tool("generate_recipe")
        async def generate_recipe(
            ingredients: List[str],
            cuisine: Optional[str] = None,
            dietary_restrictions: Optional[List[str]] = None,
            cooking_time: Optional[int] = None,
            difficulty: Optional[str] = None,
            servings: Optional[int] = None,
            meal_type: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Generate a recipe using T5-Large with RAG enhancement.
            
            Args:
                ingredients: List of ingredients to include
                cuisine: Cuisine type (e.g., "Italian", "Asian")
                dietary_restrictions: List of dietary restrictions
                cooking_time: Maximum cooking time in minutes
                difficulty: Difficulty level ("easy", "medium", "hard")
                servings: Number of servings
                meal_type: Type of meal ("breakfast", "lunch", "dinner", "snack")
                
            Returns:
                Generated recipe with metadata
            """
            return await self._generate_recipe_with_rag(
                ingredients=ingredients,
                cuisine=cuisine,
                dietary_restrictions=dietary_restrictions,
                cooking_time=cooking_time,
                difficulty=difficulty,
                servings=servings,
                meal_type=meal_type
            )
        
        @self.server.tool("refine_recipe")
        async def refine_recipe(
            original_recipe: str,
            refinement_request: str,
            context: Optional[Dict] = None
        ) -> Dict[str, Any]:
            """
            Refine an existing recipe based on user feedback.
            
            Args:
                original_recipe: The original recipe text
                refinement_request: What changes to make
                context: Additional context for refinement
                
            Returns:
                Refined recipe with change summary
            """
            return await self._refine_recipe(original_recipe, refinement_request, context)
        
        @self.server.tool("validate_recipe")
        async def validate_recipe(recipe: str) -> Dict[str, Any]:
            """
            Validate a recipe for completeness and quality.
            
            Args:
                recipe: Recipe text to validate
                
            Returns:
                Validation results with suggestions
            """
            return await self._validate_recipe(recipe)
        
        @self.server.tool("generate_variations")
        async def generate_variations(
            base_recipe: str,
            variation_type: str = "cuisine",
            count: int = 3
        ) -> List[Dict[str, Any]]:
            """
            Generate variations of a base recipe.
            
            Args:
                base_recipe: The base recipe to vary
                variation_type: Type of variation ("cuisine", "dietary", "difficulty")
                count: Number of variations to generate
                
            Returns:
                List of recipe variations
            """
            return await self._generate_variations(base_recipe, variation_type, count)
    
    async def _generate_recipe_with_rag(self, **kwargs) -> Dict[str, Any]:
        """Generate recipe using T5-Large with RAG enhancement."""
        import time
        start_time = time.time()
        
        try:
            # 1. Create search query for RAG
            search_query = self._create_search_query(kwargs)
            
            # 2. Retrieve similar recipes for context
            similar_recipes = await self.rag_system.hybrid_search(
                query=search_query,
                top_k=3,
                filters=self._create_filters(kwargs)
            )
            
            # 3. Create enhanced prompt with RAG context
            prompt = self._create_enhanced_prompt(kwargs, similar_recipes)
            
            # 4. Generate recipe with T5
            generated_text = await self._generate_with_t5(prompt)
            
            # 5. Post-process and validate
            processed_recipe = self._post_process_recipe(generated_text, kwargs)
            
            # 6. Calculate confidence score
            confidence_score = self._calculate_confidence(
                processed_recipe, 
                similar_recipes, 
                kwargs
            )
            
            # Update performance tracking
            generation_time = time.time() - start_time
            self.generation_count += 1
            self.avg_generation_time = (
                (self.avg_generation_time * (self.generation_count - 1) + generation_time) 
                / self.generation_count
            )
            
            return {
                "recipe": processed_recipe,
                "context_recipes": [
                    {
                        "title": r.get("title", "Unknown"),
                        "similarity_score": r.get("similarity_score", 0)
                    }
                    for r in similar_recipes
                ],
                "confidence_score": confidence_score,
                "generation_metadata": {
                    "model": "t5-large-recipes",
                    "rag_enhanced": True,
                    "context_count": len(similar_recipes),
                    "generation_time": round(generation_time, 3),
                    "search_query": search_query
                },
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Recipe generation failed: {e}")
            return {
                "recipe": "",
                "context_recipes": [],
                "confidence_score": 0.0,
                "generation_metadata": {
                    "model": "t5-large-recipes",
                    "rag_enhanced": False,
                    "error": str(e)
                },
                "success": False
            }
    
    def _create_search_query(self, kwargs: Dict) -> str:
        """Create search query for RAG retrieval."""
        query_parts = []
        
        # Add ingredients
        ingredients = kwargs.get("ingredients", [])
        if ingredients:
            query_parts.extend(ingredients)
        
        # Add cuisine
        cuisine = kwargs.get("cuisine")
        if cuisine:
            query_parts.append(cuisine)
        
        # Add meal type
        meal_type = kwargs.get("meal_type")
        if meal_type:
            query_parts.append(meal_type)
        
        # Add dietary restrictions
        dietary = kwargs.get("dietary_restrictions", [])
        if dietary:
            query_parts.extend(dietary)
        
        return " ".join(query_parts) if query_parts else "recipe"
    
    def _create_filters(self, kwargs: Dict) -> Dict:
        """Create filters for RAG search."""
        filters = {}
        
        if kwargs.get("cuisine"):
            filters["cuisine"] = kwargs["cuisine"]
        
        if kwargs.get("dietary_restrictions"):
            filters["dietary_restrictions"] = kwargs["dietary_restrictions"]
        
        if kwargs.get("cooking_time"):
            filters["max_cooking_time"] = kwargs["cooking_time"]
        
        if kwargs.get("difficulty"):
            filters["difficulty"] = kwargs["difficulty"]
        
        return filters
    
    def _create_enhanced_prompt(self, kwargs: Dict, similar_recipes: List[Dict]) -> str:
        """Create enhanced prompt with RAG context."""
        # Base instruction
        prompt_parts = [
            "Generate a complete recipe based on the following requirements:"
        ]
        
        # Add requirements
        requirements = []
        
        ingredients = kwargs.get("ingredients", [])
        if ingredients:
            requirements.append(f"Ingredients to include: {', '.join(ingredients)}")
        
        cuisine = kwargs.get("cuisine")
        if cuisine:
            requirements.append(f"Cuisine: {cuisine}")
        
        dietary = kwargs.get("dietary_restrictions", [])
        if dietary:
            requirements.append(f"Dietary restrictions: {', '.join(dietary)}")
        
        cooking_time = kwargs.get("cooking_time")
        if cooking_time:
            requirements.append(f"Maximum cooking time: {cooking_time} minutes")
        
        difficulty = kwargs.get("difficulty")
        if difficulty:
            requirements.append(f"Difficulty level: {difficulty}")
        
        servings = kwargs.get("servings")
        if servings:
            requirements.append(f"Servings: {servings}")
        
        meal_type = kwargs.get("meal_type")
        if meal_type:
            requirements.append(f"Meal type: {meal_type}")
        
        if requirements:
            prompt_parts.append("Requirements:")
            prompt_parts.extend([f"- {req}" for req in requirements])
        
        # Add context from similar recipes
        if similar_recipes:
            prompt_parts.append("\nSimilar recipe examples for inspiration:")
            
            for i, recipe in enumerate(similar_recipes[:2], 1):
                prompt_parts.append(f"\nExample {i}:")
                prompt_parts.append(f"Title: {recipe.get('title', 'Unknown')}")
                
                recipe_ingredients = recipe.get('ingredients', [])
                if recipe_ingredients:
                    ing_text = ", ".join(recipe_ingredients[:5])
                    prompt_parts.append(f"Key ingredients: {ing_text}")
                
                recipe_instructions = recipe.get('instructions', [])
                if recipe_instructions and len(recipe_instructions) > 0:
                    first_instruction = recipe_instructions[0] if isinstance(recipe_instructions, list) else str(recipe_instructions)
                    prompt_parts.append(f"First step: {first_instruction[:100]}...")
        
        # Add generation instruction
        prompt_parts.extend([
            "\nGenerate a complete recipe with:",
            "- Title",
            "- Ingredients list with quantities",
            "- Step-by-step instructions",
            "- Cooking time and servings",
            "\nRecipe:"
        ])
        
        return "\n".join(prompt_parts)
    
    async def _generate_with_t5(self, prompt: str) -> str:
        """Generate text using T5 model."""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0], 
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return generated_text
            
        except Exception as e:
            logger.error(f"T5 generation failed: {e}")
            return ""
    
    def _post_process_recipe(self, generated_text: str, kwargs: Dict) -> str:
        """Post-process generated recipe text."""
        # Clean up the text
        lines = generated_text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line and not line.startswith("Recipe:"):
                cleaned_lines.append(line)
        
        # Ensure proper structure
        recipe_text = '\n'.join(cleaned_lines)
        
        # Add metadata if missing
        if not any(keyword in recipe_text.lower() for keyword in ['title:', 'ingredients:', 'instructions:']):
            # Structure the recipe properly
            structured_recipe = self._structure_recipe(recipe_text, kwargs)
            return structured_recipe
        
        return recipe_text
    
    def _structure_recipe(self, text: str, kwargs: Dict) -> str:
        """Structure unstructured recipe text."""
        lines = text.split('\n')
        
        # Try to identify title (usually first line)
        title = lines[0] if lines else "Generated Recipe"
        
        # Add structure
        structured = [f"Title: {title}"]
        
        # Add servings and time if specified
        servings = kwargs.get("servings", 4)
        cooking_time = kwargs.get("cooking_time", 30)
        
        structured.append(f"Servings: {servings}")
        structured.append(f"Cooking Time: {cooking_time} minutes")
        
        # Add cuisine if specified
        cuisine = kwargs.get("cuisine")
        if cuisine:
            structured.append(f"Cuisine: {cuisine}")
        
        # Add the rest of the text as instructions
        structured.append("\nIngredients:")
        structured.append("(See generated recipe below)")
        
        structured.append("\nInstructions:")
        structured.extend(lines[1:])
        
        return '\n'.join(structured)
    
    def _calculate_confidence(self, recipe: str, context_recipes: List[Dict], kwargs: Dict) -> float:
        """Calculate confidence score for generated recipe."""
        confidence = 0.5  # Base confidence
        
        # Check if recipe has proper structure
        if all(keyword in recipe.lower() for keyword in ['ingredients', 'instructions']):
            confidence += 0.2
        
        # Check if required ingredients are mentioned
        ingredients = kwargs.get("ingredients", [])
        if ingredients:
            mentioned_count = sum(1 for ing in ingredients if ing.lower() in recipe.lower())
            ingredient_coverage = mentioned_count / len(ingredients)
            confidence += ingredient_coverage * 0.2
        
        # Boost confidence if we had good context recipes
        if context_recipes:
            avg_similarity = sum(r.get("similarity_score", 0) for r in context_recipes) / len(context_recipes)
            confidence += avg_similarity * 0.1
        
        # Check recipe length (reasonable recipes should have some length)
        if len(recipe.split()) > 50:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _refine_recipe(self, original_recipe: str, refinement_request: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Refine an existing recipe based on feedback."""
        try:
            # Create refinement prompt
            prompt = f"""Refine the following recipe based on the requested changes:

Original Recipe:
{original_recipe}

Requested Changes:
{refinement_request}

Please provide the refined recipe with the requested modifications:
"""
            
            # Generate refined version
            refined_text = await self._generate_with_t5(prompt)
            
            # Identify changes
            changes = self._identify_changes(original_recipe, refined_text)
            
            return {
                "refined_recipe": refined_text,
                "changes_made": changes,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Recipe refinement failed: {e}")
            return {
                "refined_recipe": original_recipe,
                "changes_made": [],
                "success": False,
                "error": str(e)
            }
    
    def _identify_changes(self, original: str, refined: str) -> List[str]:
        """Identify changes between original and refined recipe."""
        changes = []
        
        # Simple change detection (could be enhanced with more sophisticated NLP)
        if len(refined) > len(original) * 1.1:
            changes.append("Recipe expanded with additional details")
        elif len(refined) < len(original) * 0.9:
            changes.append("Recipe simplified and condensed")
        
        # Check for new ingredients or instructions
        original_words = set(original.lower().split())
        refined_words = set(refined.lower().split())
        
        new_words = refined_words - original_words
        if new_words:
            changes.append(f"Added new elements: {', '.join(list(new_words)[:5])}")
        
        return changes if changes else ["Recipe modified based on request"]
    
    async def _validate_recipe(self, recipe: str) -> Dict[str, Any]:
        """Validate recipe structure and content."""
        validation_results = {
            "structure_valid": False,
            "has_ingredients": False,
            "has_instructions": False,
            "reasonable_length": False,
            "has_title": False
        }
        
        recipe_lower = recipe.lower()
        
        # Check structure
        validation_results["has_title"] = any(
            keyword in recipe_lower for keyword in ['title:', 'recipe:', 'name:']
        )
        
        validation_results["has_ingredients"] = 'ingredients' in recipe_lower
        validation_results["has_instructions"] = any(
            keyword in recipe_lower for keyword in ['instructions', 'method', 'directions', 'steps']
        )
        
        validation_results["reasonable_length"] = len(recipe.split()) >= 20
        
        # Overall structure validation
        validation_results["structure_valid"] = (
            validation_results["has_ingredients"] and 
            validation_results["has_instructions"]
        )
        
        # Calculate overall score
        score = sum(validation_results.values()) / len(validation_results)
        
        # Generate suggestions
        suggestions = []
        if not validation_results["has_title"]:
            suggestions.append("Add a descriptive title")
        if not validation_results["has_ingredients"]:
            suggestions.append("Include an ingredients list")
        if not validation_results["has_instructions"]:
            suggestions.append("Add step-by-step instructions")
        if not validation_results["reasonable_length"]:
            suggestions.append("Provide more detailed instructions")
        
        return {
            "validation_results": validation_results,
            "overall_score": round(score, 2),
            "suggestions": suggestions,
            "valid": score >= 0.8
        }
    
    async def _generate_variations(self, base_recipe: str, variation_type: str, count: int) -> List[Dict[str, Any]]:
        """Generate variations of a base recipe."""
        variations = []
        
        try:
            for i in range(count):
                if variation_type == "cuisine":
                    cuisines = ["Italian", "Asian", "Mexican", "Mediterranean", "Indian"]
                    cuisine = cuisines[i % len(cuisines)]
                    prompt = f"Adapt this recipe to {cuisine} cuisine:\n{base_recipe}\n\nAdapted recipe:"
                    
                elif variation_type == "dietary":
                    dietary_options = ["vegan", "gluten-free", "low-carb", "dairy-free", "keto"]
                    dietary = dietary_options[i % len(dietary_options)]
                    prompt = f"Adapt this recipe to be {dietary}:\n{base_recipe}\n\nAdapted recipe:"
                    
                elif variation_type == "difficulty":
                    difficulty_levels = ["easier", "more advanced", "quick and simple"]
                    difficulty = difficulty_levels[i % len(difficulty_levels)]
                    prompt = f"Make this recipe {difficulty}:\n{base_recipe}\n\nModified recipe:"
                    
                else:
                    prompt = f"Create a variation of this recipe:\n{base_recipe}\n\nVariation:"
                
                # Generate variation
                variation_text = await self._generate_with_t5(prompt)
                
                variations.append({
                    "variation_type": variation_type,
                    "recipe": variation_text,
                    "variation_number": i + 1
                })
                
        except Exception as e:
            logger.error(f"Variation generation failed: {e}")
        
        return variations
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        return {
            "server_name": "recipe-generation",
            "model_loaded": self.model is not None,
            "device": self.device,
            "generation_count": self.generation_count,
            "avg_generation_time": round(self.avg_generation_time, 3),
            "max_length": self.max_length,
            "temperature": self.temperature
        }

# Server startup
async def main():
    """Start the Recipe Generation MCP Server."""
    try:
        # Initialize RAG system
        rag_system = EnhancedRAGSystem()
        
        # Initialize server
        server = RecipeGenerationServer(rag_system=rag_system)
        
        # Start server
        logger.info("Starting Recipe Generation MCP Server...")
        await server.server.run()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())