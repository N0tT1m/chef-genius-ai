"""
Enhanced RAG System with Weaviate Vector Database and Hybrid Search
Upgraded version of the existing RAG system with multi-modal capabilities
"""

import logging
import json
import asyncio
import hashlib
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import uuid

import numpy as np
import weaviate
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch

logger = logging.getLogger(__name__)

class EnhancedRAGSystem:
    """
    Enhanced RAG system with Weaviate vector database, hybrid search,
    and knowledge graph capabilities for recipe generation.
    """
    
    def __init__(self, 
                 weaviate_url: str = "http://localhost:8080",
                 embedding_model: str = "sentence-transformers/all-MiniLM-L12-v2",
                 recipe_db_path: str = None,
                 cache_size: int = 10000):
        """
        Initialize the enhanced RAG system.
        
        Args:
            weaviate_url: URL for Weaviate vector database
            embedding_model: Sentence transformer model for embeddings
            recipe_db_path: Path to recipe database
            cache_size: Size of query cache
        """
        self.weaviate_url = weaviate_url
        self.embedding_model_name = embedding_model
        self.recipe_db_path = recipe_db_path
        self.cache_size = cache_size
        
        # Initialize components
        self.client = None
        self.embedding_model = None
        self.tfidf_vectorizer = None
        self.recipe_texts = []
        self.query_cache = {}
        
        # Performance tracking
        self.query_count = 0
        self.cache_hits = 0
        self.search_times = []
        
        # Schema definitions
        self.recipe_schema = {
            "class": "Recipe",
            "description": "A cooking recipe with ingredients and instructions",
            "properties": [
                {
                    "name": "title",
                    "dataType": ["text"],
                    "description": "Recipe title"
                },
                {
                    "name": "cuisine",
                    "dataType": ["text"],
                    "description": "Cuisine type"
                },
                {
                    "name": "ingredients",
                    "dataType": ["text[]"],
                    "description": "List of ingredients"
                },
                {
                    "name": "instructions",
                    "dataType": ["text[]"],
                    "description": "Cooking instructions"
                },
                {
                    "name": "dietary_tags",
                    "dataType": ["text[]"],
                    "description": "Dietary restrictions and tags"
                },
                {
                    "name": "cooking_time",
                    "dataType": ["int"],
                    "description": "Total cooking time in minutes"
                },
                {
                    "name": "difficulty",
                    "dataType": ["text"],
                    "description": "Difficulty level"
                },
                {
                    "name": "servings",
                    "dataType": ["int"],
                    "description": "Number of servings"
                },
                {
                    "name": "searchable_text",
                    "dataType": ["text"],
                    "description": "Combined searchable text"
                },
                {
                    "name": "nutrition_info",
                    "dataType": ["object"],
                    "description": "Nutritional information"
                },
                {
                    "name": "recipe_id",
                    "dataType": ["text"],
                    "description": "Original recipe ID"
                }
            ]
        }
        
        self.ingredient_schema = {
            "class": "Ingredient",
            "description": "Cooking ingredient with properties and substitutions",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "Ingredient name"
                },
                {
                    "name": "category",
                    "dataType": ["text"],
                    "description": "Ingredient category (protein, vegetable, etc.)"
                },
                {
                    "name": "substitutions",
                    "dataType": ["text[]"],
                    "description": "Possible substitutions"
                },
                {
                    "name": "dietary_properties",
                    "dataType": ["text[]"],
                    "description": "Dietary properties (vegan, gluten-free, etc.)"
                },
                {
                    "name": "flavor_profile",
                    "dataType": ["text[]"],
                    "description": "Flavor characteristics"
                }
            ]
        }
        
        self.technique_schema = {
            "class": "Technique",
            "description": "Cooking technique or method",
            "properties": [
                {
                    "name": "name",
                    "dataType": ["text"],
                    "description": "Technique name"
                },
                {
                    "name": "description",
                    "dataType": ["text"],
                    "description": "Technique description"
                },
                {
                    "name": "difficulty",
                    "dataType": ["text"],
                    "description": "Difficulty level"
                },
                {
                    "name": "equipment_needed",
                    "dataType": ["text[]"],
                    "description": "Required equipment"
                },
                {
                    "name": "suitable_ingredients",
                    "dataType": ["text[]"],
                    "description": "Ingredients this technique works well with"
                }
            ]
        }
        
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize all system components."""
        try:
            # Connect to Weaviate
            self._connect_to_weaviate()
            
            # Load embedding model
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            
            # Initialize TF-IDF for hybrid search
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=10000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            
            # Setup schemas
            self._setup_schemas()
            
            # Load existing data if path provided
            if self.recipe_db_path:
                self._migrate_existing_data()
            
            logger.info("Enhanced RAG system initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize enhanced RAG system: {e}")
            raise
    
    def _connect_to_weaviate(self):
        """Connect to Weaviate database."""
        try:
            self.client = weaviate.Client(self.weaviate_url)
            
            # Test connection
            if self.client.is_ready():
                logger.info(f"Connected to Weaviate at {self.weaviate_url}")
            else:
                raise Exception("Weaviate is not ready")
                
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {e}")
            raise
    
    def _setup_schemas(self):
        """Setup Weaviate schemas for different data types."""
        try:
            # Create schemas if they don't exist
            schemas = [self.recipe_schema, self.ingredient_schema, self.technique_schema]
            
            for schema in schemas:
                class_name = schema["class"]
                if not self.client.schema.exists(class_name):
                    self.client.schema.create_class(schema)
                    logger.info(f"Created Weaviate schema for {class_name}")
                else:
                    logger.info(f"Schema for {class_name} already exists")
                    
        except Exception as e:
            logger.error(f"Failed to setup schemas: {e}")
            raise
    
    async def _migrate_existing_data(self):
        """Migrate data from existing RAG system to Weaviate."""
        try:
            # Load existing recipe data
            if Path(self.recipe_db_path).exists():
                with open(self.recipe_db_path, 'r', encoding='utf-8') as f:
                    recipes = json.load(f)
                    
                logger.info(f"Migrating {len(recipes)} recipes to Weaviate...")
                
                # Process in batches for efficiency
                batch_size = 100
                for i in range(0, len(recipes), batch_size):
                    batch = recipes[i:i + batch_size]
                    await self._batch_import_recipes(batch)
                    
                    if i % 1000 == 0:
                        logger.info(f"Migrated {i}/{len(recipes)} recipes")
                
                logger.info("Recipe migration completed")
                
        except Exception as e:
            logger.error(f"Failed to migrate existing data: {e}")
    
    async def _batch_import_recipes(self, recipes: List[Dict]):
        """Import a batch of recipes to Weaviate."""
        try:
            with self.client.batch as batch:
                batch.batch_size = 100
                
                for recipe in recipes:
                    # Process recipe data
                    processed_recipe = self._process_recipe_for_weaviate(recipe)
                    
                    # Add to batch
                    batch.add_data_object(
                        processed_recipe,
                        "Recipe",
                        uuid.uuid4().hex
                    )
                    
        except Exception as e:
            logger.error(f"Failed to batch import recipes: {e}")
    
    def _process_recipe_for_weaviate(self, recipe: Dict) -> Dict:
        """Process recipe data for Weaviate storage."""
        # Handle different input formats
        if "input_data" in recipe and "output_data" in recipe:
            # Training data format
            input_data = recipe["input_data"]
            output_data = recipe["output_data"]
            
            processed = {
                "title": output_data.get("title", "Untitled Recipe"),
                "cuisine": input_data.get("cuisine", "Unknown"),
                "ingredients": input_data.get("ingredients", []),
                "instructions": output_data.get("instructions", []),
                "dietary_tags": input_data.get("dietary_restrictions", []),
                "cooking_time": input_data.get("cooking_time", 30),
                "difficulty": input_data.get("difficulty", "medium"),
                "servings": input_data.get("servings", 4),
                "recipe_id": str(recipe.get("id", uuid.uuid4().hex))
            }
        else:
            # Direct format
            processed = {
                "title": recipe.get("title", "Untitled Recipe"),
                "cuisine": recipe.get("cuisine", "Unknown"),
                "ingredients": recipe.get("ingredients", []),
                "instructions": recipe.get("instructions", []),
                "dietary_tags": recipe.get("dietary_tags", []),
                "cooking_time": recipe.get("cooking_time", 30),
                "difficulty": recipe.get("difficulty", "medium"),
                "servings": recipe.get("servings", 4),
                "recipe_id": str(recipe.get("id", uuid.uuid4().hex))
            }
        
        # Create searchable text
        processed["searchable_text"] = self._create_searchable_text(processed)
        
        return processed
    
    def _create_searchable_text(self, recipe: Dict) -> str:
        """Create combined searchable text for recipe."""
        text_parts = []
        
        # Add title
        if recipe.get("title"):
            text_parts.append(f"Title: {recipe['title']}")
        
        # Add cuisine
        if recipe.get("cuisine"):
            text_parts.append(f"Cuisine: {recipe['cuisine']}")
        
        # Add ingredients
        ingredients = recipe.get("ingredients", [])
        if ingredients:
            if isinstance(ingredients, list):
                text_parts.append(f"Ingredients: {' '.join(ingredients)}")
            else:
                text_parts.append(f"Ingredients: {ingredients}")
        
        # Add instructions
        instructions = recipe.get("instructions", [])
        if instructions:
            if isinstance(instructions, list):
                text_parts.append(f"Instructions: {' '.join(instructions)}")
            else:
                text_parts.append(f"Instructions: {instructions}")
        
        # Add dietary tags
        dietary = recipe.get("dietary_tags", [])
        if dietary:
            if isinstance(dietary, list):
                text_parts.append(f"Dietary: {' '.join(dietary)}")
            else:
                text_parts.append(f"Dietary: {dietary}")
        
        return " | ".join(text_parts)
    
    async def hybrid_search(self, 
                          query: str, 
                          top_k: int = 10,
                          semantic_weight: float = 0.7,
                          keyword_weight: float = 0.3,
                          filters: Optional[Dict] = None) -> List[Dict]:
        """
        Perform hybrid search combining semantic and keyword search.
        
        Args:
            query: Search query
            top_k: Number of results to return
            semantic_weight: Weight for semantic search results
            keyword_weight: Weight for keyword search results
            filters: Additional filters to apply
            
        Returns:
            List of search results with scores
        """
        start_time = time.time()
        self.query_count += 1
        
        # Check cache
        cache_key = hashlib.md5(f"{query}_{top_k}_{filters}".encode()).hexdigest()
        if cache_key in self.query_cache:
            self.cache_hits += 1
            return self.query_cache[cache_key]
        
        try:
            # 1. Semantic search using Weaviate
            semantic_results = await self._semantic_search(query, top_k * 2, filters)
            
            # 2. Keyword search using TF-IDF (if we have recipe texts)
            keyword_results = []
            if self.recipe_texts:
                keyword_results = await self._keyword_search(query, top_k * 2, filters)
            
            # 3. Combine and re-rank results
            combined_results = self._combine_search_results(
                semantic_results, 
                keyword_results, 
                semantic_weight, 
                keyword_weight
            )
            
            # 4. Apply final ranking and filtering
            final_results = combined_results[:top_k]
            
            # Cache results
            if len(self.query_cache) < self.cache_size:
                self.query_cache[cache_key] = final_results
            
            # Track performance
            search_time = time.time() - start_time
            self.search_times.append(search_time)
            
            logger.debug(f"Hybrid search completed in {search_time:.3f}s for query: '{query}'")
            
            return final_results
            
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            return []
    
    async def _semantic_search(self, 
                             query: str, 
                             top_k: int,
                             filters: Optional[Dict] = None) -> List[Dict]:
        """Perform semantic search using Weaviate."""
        try:
            # Build the query
            where_filter = None
            if filters:
                where_filter = self._build_where_filter(filters)
            
            # Perform near text search
            result = (
                self.client.query
                .get("Recipe", [
                    "title", "cuisine", "ingredients", "instructions", 
                    "dietary_tags", "cooking_time", "difficulty", 
                    "servings", "searchable_text", "recipe_id"
                ])
                .with_near_text({"concepts": [query]})
                .with_limit(top_k)
                .with_additional(["distance"])
            )
            
            if where_filter:
                result = result.with_where(where_filter)
            
            response = result.do()
            
            # Process results
            results = []
            if "data" in response and "Get" in response["data"]:
                recipes = response["data"]["Get"].get("Recipe", [])
                
                for recipe in recipes:
                    # Convert distance to similarity score
                    distance = recipe.get("_additional", {}).get("distance", 1.0)
                    similarity = 1.0 - distance
                    
                    result_dict = {
                        "id": recipe.get("recipe_id"),
                        "title": recipe.get("title"),
                        "cuisine": recipe.get("cuisine"),
                        "ingredients": recipe.get("ingredients", []),
                        "instructions": recipe.get("instructions", []),
                        "dietary_tags": recipe.get("dietary_tags", []),
                        "cooking_time": recipe.get("cooking_time"),
                        "difficulty": recipe.get("difficulty"),
                        "servings": recipe.get("servings"),
                        "similarity_score": similarity,
                        "search_type": "semantic"
                    }
                    results.append(result_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search failed: {e}")
            return []
    
    async def _keyword_search(self, 
                            query: str, 
                            top_k: int,
                            filters: Optional[Dict] = None) -> List[Dict]:
        """Perform keyword search using TF-IDF."""
        try:
            if not self.recipe_texts:
                return []
            
            # Transform query using existing TF-IDF vectorizer
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if similarities[idx] > 0.1:  # Minimum threshold
                    # This would need to be implemented based on your data structure
                    # For now, returning placeholder
                    result = {
                        "similarity_score": float(similarities[idx]),
                        "search_type": "keyword",
                        # Add other fields as needed
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Keyword search failed: {e}")
            return []
    
    def _build_where_filter(self, filters: Dict) -> Dict:
        """Build Weaviate where filter from filter dictionary."""
        conditions = []
        
        for key, value in filters.items():
            if key == "cuisine" and value:
                conditions.append({
                    "path": ["cuisine"],
                    "operator": "Equal",
                    "valueText": value
                })
            elif key == "dietary_restrictions" and value:
                for restriction in value:
                    conditions.append({
                        "path": ["dietary_tags"],
                        "operator": "Equal",
                        "valueText": restriction
                    })
            elif key == "max_cooking_time" and value:
                conditions.append({
                    "path": ["cooking_time"],
                    "operator": "LessThanEqual",
                    "valueInt": value
                })
            elif key == "difficulty" and value:
                conditions.append({
                    "path": ["difficulty"],
                    "operator": "Equal",
                    "valueText": value
                })
        
        if len(conditions) == 1:
            return conditions[0]
        elif len(conditions) > 1:
            return {
                "operator": "And",
                "operands": conditions
            }
        
        return {}
    
    def _combine_search_results(self, 
                              semantic_results: List[Dict],
                              keyword_results: List[Dict],
                              semantic_weight: float,
                              keyword_weight: float) -> List[Dict]:
        """Combine and re-rank semantic and keyword search results."""
        # Create a dictionary to combine results by ID
        combined = {}
        
        # Add semantic results
        for result in semantic_results:
            recipe_id = result.get("id") or result.get("recipe_id", "")
            if recipe_id:
                combined[recipe_id] = result.copy()
                combined[recipe_id]["final_score"] = result["similarity_score"] * semantic_weight
        
        # Add keyword results
        for result in keyword_results:
            recipe_id = result.get("id") or result.get("recipe_id", "")
            if recipe_id:
                if recipe_id in combined:
                    # Combine scores
                    combined[recipe_id]["final_score"] += result["similarity_score"] * keyword_weight
                else:
                    # Add new result
                    combined[recipe_id] = result.copy()
                    combined[recipe_id]["final_score"] = result["similarity_score"] * keyword_weight
        
        # Sort by final score
        final_results = list(combined.values())
        final_results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        
        return final_results
    
    async def add_recipe(self, recipe: Dict) -> bool:
        """Add a new recipe to the vector database."""
        try:
            processed_recipe = self._process_recipe_for_weaviate(recipe)
            
            # Add to Weaviate
            recipe_id = self.client.data_object.create(
                processed_recipe,
                "Recipe",
                uuid.uuid4().hex
            )
            
            logger.info(f"Added recipe to Weaviate: {processed_recipe.get('title')}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add recipe: {e}")
            return False
    
    async def update_recipe(self, recipe_id: str, updates: Dict) -> bool:
        """Update an existing recipe in the database."""
        try:
            # Update in Weaviate
            self.client.data_object.update(
                updates,
                "Recipe",
                recipe_id
            )
            
            logger.info(f"Updated recipe {recipe_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update recipe {recipe_id}: {e}")
            return False
    
    async def delete_recipe(self, recipe_id: str) -> bool:
        """Delete a recipe from the database."""
        try:
            self.client.data_object.delete(recipe_id, "Recipe")
            logger.info(f"Deleted recipe {recipe_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete recipe {recipe_id}: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system performance statistics."""
        cache_hit_rate = (self.cache_hits / self.query_count * 100) if self.query_count > 0 else 0
        avg_search_time = np.mean(self.search_times) if self.search_times else 0
        
        # Get Weaviate stats
        try:
            recipe_count = self.client.query.aggregate("Recipe").with_meta_count().do()
            total_recipes = recipe_count.get("data", {}).get("Aggregate", {}).get("Recipe", [{}])[0].get("meta", {}).get("count", 0)
        except:
            total_recipes = 0
        
        return {
            "total_recipes": total_recipes,
            "embedding_model": self.embedding_model_name,
            "query_count": self.query_count,
            "cache_hits": self.cache_hits,
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "cache_size": len(self.query_cache),
            "avg_search_time": f"{avg_search_time:.3f}s",
            "weaviate_connected": self.client.is_ready() if self.client else False
        }
    
    def clear_cache(self):
        """Clear the query cache."""
        self.query_cache.clear()
        logger.info("Query cache cleared")
        
    async def health_check(self) -> Dict[str, Any]:
        """Perform system health check."""
        health = {
            "status": "healthy",
            "weaviate_connected": False,
            "embedding_model_loaded": self.embedding_model is not None,
            "schemas_exist": False
        }
        
        try:
            # Check Weaviate connection
            if self.client and self.client.is_ready():
                health["weaviate_connected"] = True
                
                # Check if schemas exist
                schema = self.client.schema.get()
                class_names = [cls["class"] for cls in schema.get("classes", [])]
                health["schemas_exist"] = all(name in class_names for name in ["Recipe", "Ingredient", "Technique"])
            
        except Exception as e:
            health["status"] = "unhealthy"
            health["error"] = str(e)
        
        return health