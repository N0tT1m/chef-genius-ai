"""
RAG (Retrieval Augmented Generation) System for Chef Genius

This module implements a sophisticated RAG system that enhances recipe generation
with knowledge retrieval from a comprehensive recipe database.
"""

import logging
import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
import hashlib
import time

logger = logging.getLogger(__name__)

class RAGSystem:
    """Advanced RAG system for recipe knowledge retrieval and generation."""
    
    def __init__(self, 
                 embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",
                 recipe_db_path: str = "/app/data/recipes.json",
                 cache_size: int = 1000):
        """
        Initialize the RAG system.
        
        Args:
            embedding_model: Name of the sentence transformer model for embeddings
            recipe_db_path: Path to the recipe database JSON file
            cache_size: Maximum number of cached embeddings
        """
        self.embedding_model_name = embedding_model
        self.recipe_db_path = recipe_db_path
        self.cache_size = cache_size
        
        # Initialize components
        self.embedding_model = None
        self.recipe_database = []
        self.recipe_embeddings = None
        self.embedding_cache = {}
        
        # Performance tracking
        self.query_count = 0
        self.cache_hits = 0
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all RAG system components."""
        try:
            # Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Load recipe database
            self._load_recipe_database()
            
            # Generate embeddings for recipes
            self._generate_recipe_embeddings()
            
            logger.info("RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            raise
    
    def _load_recipe_database(self):
        """Load the recipe database from JSON file."""
        try:
            recipe_path = Path(self.recipe_db_path)
            if not recipe_path.exists():
                # Fallback to CLI data path
                cli_path = Path("/app/cli/data/training.json")
                if cli_path.exists():
                    recipe_path = cli_path
                else:
                    logger.warning("Recipe database not found, creating sample database")
                    self._create_sample_database()
                    return
            
            with open(recipe_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
            
            # Process recipe data into standardized format
            self.recipe_database = self._process_recipe_data(raw_data)
            
            logger.info(f"Loaded {len(self.recipe_database)} recipes into RAG database")
            
        except Exception as e:
            logger.error(f"Failed to load recipe database: {e}")
            self._create_sample_database()
    
    def _process_recipe_data(self, raw_data: List[Dict]) -> List[Dict]:
        """Process raw recipe data into standardized format for RAG."""
        processed_recipes = []
        
        for i, recipe in enumerate(raw_data):
            try:
                # Handle different data formats
                if "input_data" in recipe and "output_data" in recipe:
                    # Training data format
                    input_data = recipe["input_data"]
                    output_data = recipe["output_data"]
                    
                    processed_recipe = {
                        "id": i,
                        "title": output_data.get("title", f"Recipe {i}"),
                        "ingredients": input_data.get("ingredients", []),
                        "instructions": output_data.get("instructions", []),
                        "cuisine": input_data.get("cuisine", "Unknown"),
                        "description": f"A delicious {input_data.get('cuisine', '')} recipe",
                        "dietary_tags": input_data.get("dietary_restrictions", []),
                        "searchable_text": self._create_searchable_text(input_data, output_data)
                    }
                else:
                    # Direct recipe format
                    processed_recipe = {
                        "id": i,
                        "title": recipe.get("title", f"Recipe {i}"),
                        "ingredients": recipe.get("ingredients", []),
                        "instructions": recipe.get("instructions", []),
                        "cuisine": recipe.get("cuisine", "Unknown"),
                        "description": recipe.get("description", ""),
                        "dietary_tags": recipe.get("dietary_tags", []),
                        "searchable_text": self._create_searchable_text(recipe, {})
                    }
                
                processed_recipes.append(processed_recipe)
                
            except Exception as e:
                logger.warning(f"Failed to process recipe {i}: {e}")
                continue
        
        return processed_recipes
    
    def _create_searchable_text(self, input_data: Dict, output_data: Dict) -> str:
        """Create searchable text representation of recipe."""
        text_parts = []
        
        # Add title
        title = output_data.get("title") or input_data.get("title", "")
        if title:
            text_parts.append(f"Title: {title}")
        
        # Add cuisine
        cuisine = input_data.get("cuisine") or input_data.get("cuisine", "")
        if cuisine:
            text_parts.append(f"Cuisine: {cuisine}")
        
        # Add ingredients
        ingredients = input_data.get("ingredients", [])
        if ingredients:
            ingredients_text = " ".join(ingredients) if isinstance(ingredients, list) else str(ingredients)
            text_parts.append(f"Ingredients: {ingredients_text}")
        
        # Add instructions
        instructions = output_data.get("instructions", [])
        if instructions:
            instructions_text = " ".join(instructions) if isinstance(instructions, list) else str(instructions)
            text_parts.append(f"Instructions: {instructions_text}")
        
        # Add dietary restrictions
        dietary = input_data.get("dietary_restrictions", [])
        if dietary:
            dietary_text = " ".join(dietary) if isinstance(dietary, list) else str(dietary)
            text_parts.append(f"Dietary: {dietary_text}")
        
        return " | ".join(text_parts)
    
    def _create_sample_database(self):
        """Create a sample recipe database for testing."""
        self.recipe_database = [
            {
                "id": 0,
                "title": "Classic Spaghetti Carbonara",
                "ingredients": ["spaghetti", "eggs", "parmesan cheese", "pancetta", "black pepper"],
                "instructions": ["Cook pasta", "Mix eggs and cheese", "Combine with pasta"],
                "cuisine": "Italian",
                "description": "Traditional Italian pasta with eggs and cheese",
                "dietary_tags": [],
                "searchable_text": "Italian pasta carbonara spaghetti eggs cheese pancetta"
            },
            {
                "id": 1,
                "title": "Vegetable Stir Fry",
                "ingredients": ["mixed vegetables", "soy sauce", "ginger", "garlic", "oil"],
                "instructions": ["Heat oil", "Add vegetables", "Stir fry with sauce"],
                "cuisine": "Asian",
                "description": "Quick and healthy vegetable stir fry",
                "dietary_tags": ["vegetarian", "vegan"],
                "searchable_text": "Asian stir fry vegetables vegetarian vegan healthy"
            }
        ]
        logger.info("Created sample recipe database")
    
    def _generate_recipe_embeddings(self):
        """Generate embeddings for all recipes in the database."""
        if not self.recipe_database:
            logger.warning("No recipes in database to embed")
            return
        
        try:
            # Extract searchable text from all recipes
            recipe_texts = [recipe["searchable_text"] for recipe in self.recipe_database]
            
            # Generate embeddings
            logger.info(f"Generating embeddings for {len(recipe_texts)} recipes...")
            self.recipe_embeddings = self.embedding_model.encode(
                recipe_texts,
                convert_to_tensor=True,
                show_progress_bar=True
            )
            
            logger.info(f"Generated embeddings shape: {self.recipe_embeddings.shape}")
            
        except Exception as e:
            logger.error(f"Failed to generate recipe embeddings: {e}")
            raise
    
    async def search_similar_recipes(self, 
                                   query: str, 
                                   top_k: int = 5,
                                   min_similarity: float = 0.3) -> List[Dict]:
        """
        Search for recipes similar to the query.
        
        Args:
            query: Search query (ingredients, cuisine, dish type, etc.)
            top_k: Number of top results to return
            min_similarity: Minimum similarity score to include
            
        Returns:
            List of similar recipes with similarity scores
        """
        if not self.recipe_embeddings is not None:
            logger.warning("No recipe embeddings available")
            return []
        
        try:
            self.query_count += 1
            
            # Check cache first
            query_hash = hashlib.md5(query.encode()).hexdigest()
            if query_hash in self.embedding_cache:
                query_embedding = self.embedding_cache[query_hash]
                self.cache_hits += 1
            else:
                # Generate query embedding
                query_embedding = self.embedding_model.encode([query], convert_to_tensor=True)
                
                # Cache the embedding
                if len(self.embedding_cache) < self.cache_size:
                    self.embedding_cache[query_hash] = query_embedding
            
            # Calculate similarities
            similarities = torch.cosine_similarity(
                query_embedding, 
                self.recipe_embeddings, 
                dim=1
            ).cpu().numpy()
            
            # Get top-k similar recipes
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                similarity_score = float(similarities[idx])
                if similarity_score >= min_similarity:
                    recipe = self.recipe_database[int(idx)].copy()
                    recipe["similarity_score"] = similarity_score
                    results.append(recipe)
            
            logger.debug(f"Found {len(results)} similar recipes for query: '{query}'")
            return results
            
        except Exception as e:
            logger.error(f"Recipe search failed: {e}")
            return []
    
    async def generate_rag_enhanced_recipe(self, 
                                         request,
                                         llm_pipeline,
                                         context_recipes: int = 3) -> Dict[str, Any]:
        """
        Generate a recipe using RAG-enhanced context.
        
        Args:
            request: Recipe generation request
            llm_pipeline: Language model pipeline for generation
            context_recipes: Number of similar recipes to use as context
            
        Returns:
            Generated recipe with metadata
        """
        try:
            # Create search query from request
            search_query = self._create_search_query(request)
            
            # Retrieve similar recipes
            similar_recipes = await self.search_similar_recipes(
                search_query, 
                top_k=context_recipes,
                min_similarity=0.4
            )
            
            # Create enhanced prompt with retrieved context
            enhanced_prompt = self._create_rag_prompt(request, similar_recipes)
            
            # Generate recipe with enhanced context
            generated_text = await self._generate_with_context(enhanced_prompt, llm_pipeline)
            
            return {
                "generated_text": generated_text,
                "context_recipes": similar_recipes,
                "search_query": search_query,
                "rag_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"RAG-enhanced generation failed: {e}")
            return {
                "generated_text": "",
                "context_recipes": [],
                "search_query": "",
                "rag_enhanced": False,
                "error": str(e)
            }
    
    def _create_search_query(self, request) -> str:
        """Create search query from recipe generation request."""
        query_parts = []
        
        if hasattr(request, 'ingredients') and request.ingredients:
            query_parts.append(" ".join(request.ingredients))
        
        if hasattr(request, 'cuisine') and request.cuisine:
            query_parts.append(request.cuisine)
        
        if hasattr(request, 'meal_type') and request.meal_type:
            query_parts.append(request.meal_type)
        
        if hasattr(request, 'dietary_restrictions') and request.dietary_restrictions:
            query_parts.extend(request.dietary_restrictions)
        
        return " ".join(query_parts) or "general recipe"
    
    def _create_rag_prompt(self, request, similar_recipes: List[Dict]) -> str:
        """Create enhanced prompt with retrieved recipe context."""
        # Base system prompt
        system_prompt = """You are an expert chef AI. Use the provided example recipes as inspiration to create a new, original recipe that meets the user's requirements. Don't copy exactly, but learn from the techniques, ingredient combinations, and cooking methods shown in the examples."""
        
        # Add context from similar recipes
        context_section = ""
        if similar_recipes:
            context_section = "\n\nSimilar recipe examples for inspiration:\n"
            for i, recipe in enumerate(similar_recipes[:3], 1):
                context_section += f"\nExample {i} ({recipe.get('similarity_score', 0):.2f} similarity):\n"
                context_section += f"Title: {recipe.get('title', 'Unknown')}\n"
                context_section += f"Cuisine: {recipe.get('cuisine', 'Unknown')}\n"
                
                ingredients = recipe.get('ingredients', [])
                if ingredients:
                    ing_text = ", ".join(ingredients[:8])  # Limit to first 8 ingredients
                    context_section += f"Key ingredients: {ing_text}\n"
                
                instructions = recipe.get('instructions', [])
                if instructions:
                    # Take first 2 instructions as technique examples
                    inst_text = ". ".join(instructions[:2])
                    context_section += f"Technique example: {inst_text}\n"
        
        # Create user request
        requirements = []
        if hasattr(request, 'ingredients') and request.ingredients:
            requirements.append(f"Must include: {', '.join(request.ingredients)}")
        if hasattr(request, 'cuisine') and request.cuisine:
            requirements.append(f"Cuisine: {request.cuisine}")
        if hasattr(request, 'dietary_restrictions') and request.dietary_restrictions:
            requirements.append(f"Dietary requirements: {', '.join(request.dietary_restrictions)}")
        if hasattr(request, 'cooking_time') and request.cooking_time:
            requirements.append(f"Cooking time: {request.cooking_time}")
        if hasattr(request, 'difficulty') and request.difficulty:
            requirements.append(f"Difficulty: {request.difficulty}")
        
        user_request = "Create an original recipe with these requirements:\n" + "\n".join(f"- {req}" for req in requirements)
        user_request += "\n\nProvide a complete recipe with title, ingredients list with amounts, and detailed step-by-step instructions."
        
        # Combine into full prompt
        full_prompt = f"{system_prompt}{context_section}\n\nUser Request:\n{user_request}\n\nRecipe:"
        
        return full_prompt
    
    async def _generate_with_context(self, prompt: str, llm_pipeline) -> str:
        """Generate text using the language model with RAG context."""
        try:
            result = llm_pipeline(
                prompt,
                max_new_tokens=1024,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                return_full_text=False
            )
            
            return result[0]['generated_text']
            
        except Exception as e:
            logger.error(f"Context-aware generation failed: {e}")
            return ""
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get RAG system performance statistics."""
        cache_hit_rate = (self.cache_hits / self.query_count * 100) if self.query_count > 0 else 0
        
        return {
            "total_recipes": len(self.recipe_database),
            "embedding_model": self.embedding_model_name,
            "query_count": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self.embedding_cache),
            "embeddings_generated": self.recipe_embeddings is not None
        }
    
    async def add_recipe_to_database(self, recipe: Dict) -> bool:
        """Add a new recipe to the database and update embeddings."""
        try:
            # Add ID
            recipe["id"] = len(self.recipe_database)
            
            # Create searchable text
            recipe["searchable_text"] = self._create_searchable_text(recipe, recipe)
            
            # Add to database
            self.recipe_database.append(recipe)
            
            # Update embeddings
            new_embedding = self.embedding_model.encode([recipe["searchable_text"]], convert_to_tensor=True)
            
            if self.recipe_embeddings is not None:
                self.recipe_embeddings = torch.cat([self.recipe_embeddings, new_embedding], dim=0)
            else:
                self.recipe_embeddings = new_embedding
            
            logger.info(f"Added new recipe to RAG database: {recipe.get('title', 'Unknown')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add recipe to database: {e}")
            return False
    
    def clear_cache(self):
        """Clear the embedding cache."""
        self.embedding_cache.clear()
        logger.info("Embedding cache cleared")