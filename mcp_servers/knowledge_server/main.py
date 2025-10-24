"""
Knowledge Retrieval MCP Server
Handles advanced search, ingredient substitutions, and cooking techniques
"""

import asyncio
import json
import logging
import os
import sys
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from collections import defaultdict
import networkx as nx

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from mcp.server import Server
from mcp.types import Tool, TextContent
from backend.app.services.enhanced_rag_system import EnhancedRAGSystem

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KnowledgeRetrievalServer:
    """MCP Server for knowledge retrieval and culinary intelligence."""
    
    def __init__(self, rag_system: EnhancedRAGSystem = None):
        """
        Initialize the knowledge retrieval server.
        
        Args:
            rag_system: Enhanced RAG system instance
        """
        self.server = Server("knowledge-retrieval")
        self.rag_system = rag_system or EnhancedRAGSystem()
        
        # Knowledge graph for ingredient relationships
        self.ingredient_graph = nx.Graph()
        self.technique_db = {}
        self.substitution_db = {}
        self.flavor_profiles = {}
        
        # Performance tracking
        self.search_count = 0
        self.substitution_requests = 0
        
        self._setup_tools()
        self._initialize_knowledge_base()
    
    def _initialize_knowledge_base(self):
        """Initialize the culinary knowledge base."""
        try:
            # Load ingredient relationships
            self._build_ingredient_graph()
            
            # Load cooking techniques
            self._load_cooking_techniques()
            
            # Load substitution mappings
            self._load_substitution_mappings()
            
            # Load flavor profiles
            self._load_flavor_profiles()
            
            logger.info("Knowledge base initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base: {e}")
    
    def _build_ingredient_graph(self):
        """Build ingredient relationship graph."""
        # Define ingredient categories and relationships
        ingredient_categories = {
            "proteins": ["chicken", "beef", "pork", "fish", "tofu", "eggs", "beans", "lentils"],
            "vegetables": ["onion", "garlic", "tomato", "carrot", "potato", "spinach", "broccoli"],
            "grains": ["rice", "pasta", "quinoa", "bread", "oats", "barley"],
            "dairy": ["milk", "cheese", "yogurt", "butter", "cream"],
            "spices": ["salt", "pepper", "cumin", "paprika", "oregano", "basil", "thyme"],
            "oils": ["olive oil", "vegetable oil", "coconut oil", "butter"]
        }
        
        # Add nodes
        for category, ingredients in ingredient_categories.items():
            for ingredient in ingredients:
                self.ingredient_graph.add_node(
                    ingredient, 
                    category=category,
                    substitutable_with=[]
                )
        
        # Add edges for similar ingredients
        substitution_pairs = [
            ("chicken", "turkey"), ("beef", "lamb"), ("pork", "chicken"),
            ("milk", "almond milk"), ("butter", "olive oil"), 
            ("rice", "quinoa"), ("pasta", "rice"),
            ("onion", "shallot"), ("garlic", "garlic powder"),
            ("oregano", "basil"), ("cumin", "coriander")
        ]
        
        for ingredient1, ingredient2 in substitution_pairs:
            if ingredient1 in self.ingredient_graph and ingredient2 in self.ingredient_graph:
                self.ingredient_graph.add_edge(ingredient1, ingredient2, 
                                             relationship="substitutable")
    
    def _load_cooking_techniques(self):
        """Load cooking technique database."""
        self.technique_db = {
            "sautéing": {
                "description": "Quick cooking in a small amount of oil or fat over high heat",
                "suitable_for": ["vegetables", "proteins", "aromatics"],
                "equipment": ["pan", "spatula"],
                "difficulty": "easy",
                "time_range": "2-10 minutes"
            },
            "braising": {
                "description": "Slow cooking in liquid after initial browning",
                "suitable_for": ["tough cuts of meat", "root vegetables"],
                "equipment": ["dutch oven", "heavy pot"],
                "difficulty": "medium",
                "time_range": "1-4 hours"
            },
            "roasting": {
                "description": "Dry heat cooking in an oven",
                "suitable_for": ["whole proteins", "vegetables", "nuts"],
                "equipment": ["oven", "roasting pan"],
                "difficulty": "easy",
                "time_range": "20 minutes - 3 hours"
            },
            "steaming": {
                "description": "Cooking with moist heat from steam",
                "suitable_for": ["vegetables", "fish", "dumplings"],
                "equipment": ["steamer", "pot with lid"],
                "difficulty": "easy",
                "time_range": "5-20 minutes"
            },
            "grilling": {
                "description": "Cooking over direct heat source",
                "suitable_for": ["proteins", "vegetables", "fruits"],
                "equipment": ["grill", "tongs"],
                "difficulty": "medium",
                "time_range": "5-30 minutes"
            }
        }
    
    def _load_substitution_mappings(self):
        """Load ingredient substitution mappings."""
        self.substitution_db = {
            "eggs": [
                {"substitute": "flax eggs", "ratio": "1:1", "dietary": ["vegan"]},
                {"substitute": "applesauce", "ratio": "1/4 cup per egg", "dietary": ["vegan"]},
                {"substitute": "banana", "ratio": "1/4 cup mashed per egg", "dietary": ["vegan"]}
            ],
            "butter": [
                {"substitute": "olive oil", "ratio": "3/4 amount", "dietary": ["dairy-free"]},
                {"substitute": "coconut oil", "ratio": "1:1", "dietary": ["dairy-free", "vegan"]},
                {"substitute": "applesauce", "ratio": "1/2 amount", "dietary": ["low-fat", "vegan"]}
            ],
            "milk": [
                {"substitute": "almond milk", "ratio": "1:1", "dietary": ["dairy-free", "vegan"]},
                {"substitute": "oat milk", "ratio": "1:1", "dietary": ["dairy-free", "vegan"]},
                {"substitute": "coconut milk", "ratio": "1:1", "dietary": ["dairy-free", "vegan"]}
            ],
            "flour": [
                {"substitute": "almond flour", "ratio": "1:1", "dietary": ["gluten-free", "low-carb"]},
                {"substitute": "coconut flour", "ratio": "1/4 amount", "dietary": ["gluten-free", "low-carb"]},
                {"substitute": "oat flour", "ratio": "1:1", "dietary": ["gluten-free"]}
            ],
            "sugar": [
                {"substitute": "honey", "ratio": "3/4 amount", "dietary": ["natural"]},
                {"substitute": "maple syrup", "ratio": "3/4 amount", "dietary": ["vegan", "natural"]},
                {"substitute": "stevia", "ratio": "1/3 amount", "dietary": ["low-calorie", "diabetic"]}
            ]
        }
    
    def _load_flavor_profiles(self):
        """Load flavor profile database."""
        self.flavor_profiles = {
            "italian": {
                "key_ingredients": ["tomato", "basil", "oregano", "garlic", "olive oil", "parmesan"],
                "techniques": ["sautéing", "roasting", "braising"],
                "characteristics": ["herbaceous", "savory", "rich"]
            },
            "asian": {
                "key_ingredients": ["ginger", "garlic", "soy sauce", "sesame oil", "rice", "vegetables"],
                "techniques": ["stir-frying", "steaming", "braising"],
                "characteristics": ["umami", "balanced", "fresh"]
            },
            "mexican": {
                "key_ingredients": ["cumin", "chili", "lime", "cilantro", "beans", "corn"],
                "techniques": ["grilling", "roasting", "braising"],
                "characteristics": ["spicy", "bright", "robust"]
            },
            "mediterranean": {
                "key_ingredients": ["olive oil", "lemon", "herbs", "vegetables", "fish", "grains"],
                "techniques": ["grilling", "roasting", "sautéing"],
                "characteristics": ["fresh", "light", "herbaceous"]
            }
        }
    
    def _setup_tools(self):
        """Setup MCP tools for knowledge retrieval."""
        
        @self.server.tool("search_recipes")
        async def search_recipes(
            query: str,
            cuisine: Optional[str] = None,
            dietary_restrictions: Optional[List[str]] = None,
            max_cooking_time: Optional[int] = None,
            difficulty: Optional[str] = None,
            top_k: int = 5
        ) -> List[Dict[str, Any]]:
            """
            Search recipe database with advanced filtering.
            
            Args:
                query: Search query
                cuisine: Cuisine filter
                dietary_restrictions: Dietary restriction filters
                max_cooking_time: Maximum cooking time filter
                difficulty: Difficulty level filter
                top_k: Number of results to return
                
            Returns:
                List of matching recipes
            """
            return await self._search_recipes(
                query, cuisine, dietary_restrictions, 
                max_cooking_time, difficulty, top_k
            )
        
        @self.server.tool("get_ingredient_substitutions")
        async def get_ingredient_substitutions(
            ingredient: str,
            dietary_restrictions: Optional[List[str]] = None,
            recipe_context: Optional[str] = None
        ) -> List[Dict[str, Any]]:
            """
            Get substitution suggestions for ingredients.
            
            Args:
                ingredient: Ingredient to find substitutions for
                dietary_restrictions: Dietary restrictions to consider
                recipe_context: Context of the recipe for better suggestions
                
            Returns:
                List of substitution options
            """
            return await self._get_substitutions(ingredient, dietary_restrictions, recipe_context)
        
        @self.server.tool("get_cooking_techniques")
        async def get_cooking_techniques(
            ingredient: str,
            cuisine: Optional[str] = None,
            difficulty: Optional[str] = None
        ) -> List[Dict[str, Any]]:
            """
            Get cooking techniques for specific ingredients.
            
            Args:
                ingredient: Ingredient to get techniques for
                cuisine: Cuisine context
                difficulty: Preferred difficulty level
                
            Returns:
                List of suitable cooking techniques
            """
            return await self._get_techniques(ingredient, cuisine, difficulty)
        
        @self.server.tool("analyze_flavor_profile")
        async def analyze_flavor_profile(
            ingredients: List[str],
            cuisine: Optional[str] = None
        ) -> Dict[str, Any]:
            """
            Analyze flavor profile of ingredient combination.
            
            Args:
                ingredients: List of ingredients to analyze
                cuisine: Cuisine context for analysis
                
            Returns:
                Flavor profile analysis
            """
            return await self._analyze_flavor_profile(ingredients, cuisine)
        
        @self.server.tool("suggest_ingredient_pairings")
        async def suggest_ingredient_pairings(
            base_ingredient: str,
            cuisine: Optional[str] = None,
            count: int = 5
        ) -> List[Dict[str, Any]]:
            """
            Suggest ingredients that pair well with a base ingredient.
            
            Args:
                base_ingredient: Base ingredient to find pairings for
                cuisine: Cuisine context
                count: Number of suggestions to return
                
            Returns:
                List of ingredient pairing suggestions
            """
            return await self._suggest_pairings(base_ingredient, cuisine, count)
        
        @self.server.tool("get_recipe_variations")
        async def get_recipe_variations(
            base_recipe: str,
            variation_type: str = "cuisine",
            target_cuisine: Optional[str] = None,
            dietary_target: Optional[str] = None
        ) -> List[Dict[str, Any]]:
            """
            Get variations of a base recipe.
            
            Args:
                base_recipe: Base recipe text or title
                variation_type: Type of variation (cuisine, dietary, technique)
                target_cuisine: Target cuisine for variation
                dietary_target: Target dietary restriction
                
            Returns:
                List of recipe variations
            """
            return await self._get_variations(base_recipe, variation_type, target_cuisine, dietary_target)
    
    async def _search_recipes(self, query: str, cuisine: Optional[str], 
                            dietary_restrictions: Optional[List[str]], 
                            max_cooking_time: Optional[int], 
                            difficulty: Optional[str], top_k: int) -> List[Dict[str, Any]]:
        """Search recipes with advanced filtering."""
        self.search_count += 1
        
        try:
            # Build filters
            filters = {}
            if cuisine:
                filters["cuisine"] = cuisine
            if dietary_restrictions:
                filters["dietary_restrictions"] = dietary_restrictions
            if max_cooking_time:
                filters["max_cooking_time"] = max_cooking_time
            if difficulty:
                filters["difficulty"] = difficulty
            
            # Perform hybrid search
            results = await self.rag_system.hybrid_search(
                query=query,
                top_k=top_k,
                filters=filters
            )
            
            # Enhance results with additional metadata
            enhanced_results = []
            for result in results:
                enhanced_result = result.copy()
                
                # Add technique suggestions
                ingredients = result.get("ingredients", [])
                if ingredients:
                    techniques = await self._suggest_techniques_for_ingredients(ingredients)
                    enhanced_result["suggested_techniques"] = techniques[:3]
                
                # Add substitution info for key ingredients
                if ingredients:
                    substitution_info = {}
                    for ingredient in ingredients[:3]:  # Limit to first 3 ingredients
                        subs = await self._get_substitutions(ingredient, dietary_restrictions)
                        if subs:
                            substitution_info[ingredient] = subs[:2]  # Top 2 substitutions
                    enhanced_result["substitution_options"] = substitution_info
                
                enhanced_results.append(enhanced_result)
            
            return enhanced_results
            
        except Exception as e:
            logger.error(f"Recipe search failed: {e}")
            return []
    
    async def _get_substitutions(self, ingredient: str, dietary_restrictions: Optional[List[str]], 
                               recipe_context: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get ingredient substitutions."""
        self.substitution_requests += 1
        
        try:
            substitutions = []
            
            # Check direct substitution database
            ingredient_lower = ingredient.lower()
            if ingredient_lower in self.substitution_db:
                for sub in self.substitution_db[ingredient_lower]:
                    # Filter by dietary restrictions
                    if dietary_restrictions:
                        if any(diet in sub.get("dietary", []) for diet in dietary_restrictions):
                            substitutions.append(sub)
                    else:
                        substitutions.append(sub)
            
            # Check ingredient graph for network-based substitutions
            if ingredient_lower in self.ingredient_graph:
                neighbors = list(self.ingredient_graph.neighbors(ingredient_lower))
                for neighbor in neighbors:
                    relationship = self.ingredient_graph[ingredient_lower][neighbor].get("relationship")
                    if relationship == "substitutable":
                        substitutions.append({
                            "substitute": neighbor,
                            "ratio": "1:1",
                            "dietary": [],
                            "source": "network"
                        })
            
            # Use RAG to find recipe-based substitutions
            if recipe_context:
                rag_substitutions = await self._find_rag_substitutions(
                    ingredient, recipe_context, dietary_restrictions
                )
                substitutions.extend(rag_substitutions)
            
            # Remove duplicates and rank by relevance
            unique_substitutions = self._deduplicate_substitutions(substitutions)
            ranked_substitutions = self._rank_substitutions(unique_substitutions, dietary_restrictions)
            
            return ranked_substitutions[:5]  # Return top 5
            
        except Exception as e:
            logger.error(f"Substitution lookup failed: {e}")
            return []
    
    async def _find_rag_substitutions(self, ingredient: str, recipe_context: str, 
                                    dietary_restrictions: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Find substitutions using RAG search."""
        try:
            # Search for recipes that use substitutions for this ingredient
            query = f"{ingredient} substitute alternative replacement"
            if dietary_restrictions:
                query += f" {' '.join(dietary_restrictions)}"
            
            results = await self.rag_system.hybrid_search(
                query=query,
                top_k=10,
                filters={"dietary_restrictions": dietary_restrictions} if dietary_restrictions else {}
            )
            
            # Extract substitution patterns from results
            substitutions = []
            for result in results:
                # This would require more sophisticated NLP to extract substitution patterns
                # For now, return placeholder structure
                ingredients = result.get("ingredients", [])
                for ing in ingredients:
                    if ing.lower() != ingredient.lower() and self._is_similar_ingredient(ingredient, ing):
                        substitutions.append({
                            "substitute": ing,
                            "ratio": "1:1",
                            "dietary": result.get("dietary_tags", []),
                            "source": "rag",
                            "confidence": result.get("similarity_score", 0.5)
                        })
            
            return substitutions
            
        except Exception as e:
            logger.error(f"RAG substitution search failed: {e}")
            return []
    
    def _is_similar_ingredient(self, ingredient1: str, ingredient2: str) -> bool:
        """Check if two ingredients are similar (simple heuristic)."""
        # Simple similarity check - could be enhanced with more sophisticated matching
        ingredient1_tokens = set(ingredient1.lower().split())
        ingredient2_tokens = set(ingredient2.lower().split())
        
        # Check for partial matches
        overlap = ingredient1_tokens.intersection(ingredient2_tokens)
        return len(overlap) > 0 or ingredient1.lower() in ingredient2.lower() or ingredient2.lower() in ingredient1.lower()
    
    def _deduplicate_substitutions(self, substitutions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate substitutions."""
        seen = set()
        unique = []
        
        for sub in substitutions:
            substitute_name = sub.get("substitute", "").lower()
            if substitute_name not in seen:
                seen.add(substitute_name)
                unique.append(sub)
        
        return unique
    
    def _rank_substitutions(self, substitutions: List[Dict[str, Any]], 
                          dietary_restrictions: Optional[List[str]]) -> List[Dict[str, Any]]:
        """Rank substitutions by relevance."""
        def score_substitution(sub):
            score = 0
            
            # Boost score for dietary match
            if dietary_restrictions:
                sub_dietary = sub.get("dietary", [])
                if any(diet in sub_dietary for diet in dietary_restrictions):
                    score += 2
            
            # Boost score for confidence
            score += sub.get("confidence", 0.5)
            
            # Boost score for direct database matches
            if sub.get("source") != "rag":
                score += 1
            
            return score
        
        return sorted(substitutions, key=score_substitution, reverse=True)
    
    async def _get_techniques(self, ingredient: str, cuisine: Optional[str], 
                            difficulty: Optional[str]) -> List[Dict[str, Any]]:
        """Get cooking techniques for ingredient."""
        try:
            suitable_techniques = []
            
            # Get ingredient category
            ingredient_category = self._get_ingredient_category(ingredient)
            
            # Find suitable techniques from database
            for technique_name, technique_info in self.technique_db.items():
                if ingredient_category in technique_info.get("suitable_for", []):
                    # Filter by difficulty if specified
                    if difficulty and technique_info.get("difficulty") != difficulty:
                        continue
                    
                    technique_result = technique_info.copy()
                    technique_result["name"] = technique_name
                    suitable_techniques.append(technique_result)
            
            # Enhance with cuisine-specific techniques
            if cuisine and cuisine.lower() in self.flavor_profiles:
                cuisine_techniques = self.flavor_profiles[cuisine.lower()].get("techniques", [])
                for technique in suitable_techniques:
                    if technique["name"] in cuisine_techniques:
                        technique["cuisine_recommended"] = True
            
            # Use RAG to find technique examples
            for technique in suitable_techniques:
                examples = await self._find_technique_examples(technique["name"], ingredient)
                technique["recipe_examples"] = examples[:2]
            
            return suitable_techniques
            
        except Exception as e:
            logger.error(f"Technique lookup failed: {e}")
            return []
    
    def _get_ingredient_category(self, ingredient: str) -> str:
        """Get category for an ingredient."""
        ingredient_lower = ingredient.lower()
        
        # Check ingredient graph
        if ingredient_lower in self.ingredient_graph:
            return self.ingredient_graph.nodes[ingredient_lower].get("category", "unknown")
        
        # Simple categorization based on keywords
        if any(protein in ingredient_lower for protein in ["chicken", "beef", "fish", "meat", "pork"]):
            return "proteins"
        elif any(veg in ingredient_lower for veg in ["vegetable", "onion", "carrot", "tomato"]):
            return "vegetables"
        elif any(grain in ingredient_lower for grain in ["rice", "pasta", "bread", "grain"]):
            return "grains"
        else:
            return "unknown"
    
    async def _find_technique_examples(self, technique: str, ingredient: str) -> List[Dict[str, Any]]:
        """Find recipe examples using a specific technique with an ingredient."""
        try:
            query = f"{technique} {ingredient} recipe"
            results = await self.rag_system.hybrid_search(query=query, top_k=3)
            
            examples = []
            for result in results:
                examples.append({
                    "title": result.get("title", "Unknown"),
                    "similarity_score": result.get("similarity_score", 0)
                })
            
            return examples
            
        except Exception as e:
            logger.error(f"Technique example search failed: {e}")
            return []
    
    async def _suggest_techniques_for_ingredients(self, ingredients: List[str]) -> List[str]:
        """Suggest techniques for a list of ingredients."""
        technique_scores = defaultdict(int)
        
        for ingredient in ingredients:
            ingredient_category = self._get_ingredient_category(ingredient)
            
            for technique_name, technique_info in self.technique_db.items():
                if ingredient_category in technique_info.get("suitable_for", []):
                    technique_scores[technique_name] += 1
        
        # Sort by score and return top techniques
        sorted_techniques = sorted(technique_scores.items(), key=lambda x: x[1], reverse=True)
        return [technique for technique, score in sorted_techniques]
    
    async def _analyze_flavor_profile(self, ingredients: List[str], cuisine: Optional[str]) -> Dict[str, Any]:
        """Analyze flavor profile of ingredient combination."""
        try:
            analysis = {
                "primary_flavors": [],
                "cuisine_match": None,
                "balance_score": 0.5,
                "suggestions": []
            }
            
            # Simple flavor analysis based on ingredient categories
            categories = [self._get_ingredient_category(ing) for ing in ingredients]
            category_counts = defaultdict(int)
            
            for category in categories:
                category_counts[category] += 1
            
            # Determine primary flavors based on categories
            if category_counts["spices"] > 0:
                analysis["primary_flavors"].append("aromatic")
            if category_counts["proteins"] > 0:
                analysis["primary_flavors"].append("savory")
            if category_counts["vegetables"] > 0:
                analysis["primary_flavors"].append("fresh")
            
            # Check cuisine match
            if cuisine and cuisine.lower() in self.flavor_profiles:
                profile = self.flavor_profiles[cuisine.lower()]
                key_ingredients = profile["key_ingredients"]
                
                # Calculate match score
                matches = sum(1 for ing in ingredients if any(key in ing.lower() for key in key_ingredients))
                match_score = matches / len(ingredients) if ingredients else 0
                
                analysis["cuisine_match"] = {
                    "cuisine": cuisine,
                    "match_score": round(match_score, 2),
                    "missing_elements": [key for key in key_ingredients[:3] if not any(key in ing.lower() for ing in ingredients)]
                }
            
            # Balance analysis
            balance_score = min(len(set(categories)) / 4, 1.0)  # Higher score for more diverse categories
            analysis["balance_score"] = round(balance_score, 2)
            
            # Suggestions
            if balance_score < 0.5:
                analysis["suggestions"].append("Consider adding ingredients from different categories for better balance")
            
            if cuisine and analysis["cuisine_match"] and analysis["cuisine_match"]["match_score"] < 0.3:
                missing = analysis["cuisine_match"]["missing_elements"]
                if missing:
                    analysis["suggestions"].append(f"Consider adding {', '.join(missing[:2])} for authentic {cuisine} flavor")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Flavor profile analysis failed: {e}")
            return {"error": str(e)}
    
    async def _suggest_pairings(self, base_ingredient: str, cuisine: Optional[str], count: int) -> List[Dict[str, Any]]:
        """Suggest ingredient pairings."""
        try:
            pairings = []
            
            # Use RAG to find common pairings
            query = f"{base_ingredient} pairs well with goes with combines"
            if cuisine:
                query += f" {cuisine}"
            
            results = await self.rag_system.hybrid_search(query=query, top_k=10)
            
            # Extract common ingredients from results
            ingredient_counts = defaultdict(int)
            
            for result in results:
                ingredients = result.get("ingredients", [])
                for ingredient in ingredients:
                    if ingredient.lower() != base_ingredient.lower():
                        ingredient_counts[ingredient] += 1
            
            # Sort by frequency and create pairing suggestions
            sorted_ingredients = sorted(ingredient_counts.items(), key=lambda x: x[1], reverse=True)
            
            for ingredient, frequency in sorted_ingredients[:count]:
                pairing = {
                    "ingredient": ingredient,
                    "frequency": frequency,
                    "compatibility_score": min(frequency / len(results), 1.0),
                    "suggested_techniques": await self._get_shared_techniques(base_ingredient, ingredient)
                }
                pairings.append(pairing)
            
            return pairings
            
        except Exception as e:
            logger.error(f"Pairing suggestion failed: {e}")
            return []
    
    async def _get_shared_techniques(self, ingredient1: str, ingredient2: str) -> List[str]:
        """Get cooking techniques that work well for both ingredients."""
        techniques1 = set()
        techniques2 = set()
        
        category1 = self._get_ingredient_category(ingredient1)
        category2 = self._get_ingredient_category(ingredient2)
        
        for technique_name, technique_info in self.technique_db.items():
            suitable_for = technique_info.get("suitable_for", [])
            if category1 in suitable_for:
                techniques1.add(technique_name)
            if category2 in suitable_for:
                techniques2.add(technique_name)
        
        shared_techniques = list(techniques1.intersection(techniques2))
        return shared_techniques[:3]
    
    async def _get_variations(self, base_recipe: str, variation_type: str, 
                            target_cuisine: Optional[str], dietary_target: Optional[str]) -> List[Dict[str, Any]]:
        """Get recipe variations."""
        try:
            variations = []
            
            if variation_type == "cuisine" and target_cuisine:
                # Find similar recipes in target cuisine
                query = f"{base_recipe} {target_cuisine} style"
                results = await self.rag_system.hybrid_search(
                    query=query,
                    top_k=5,
                    filters={"cuisine": target_cuisine}
                )
                
                for result in results:
                    variations.append({
                        "type": "cuisine_variation",
                        "target_cuisine": target_cuisine,
                        "recipe": result,
                        "similarity_to_base": result.get("similarity_score", 0)
                    })
            
            elif variation_type == "dietary" and dietary_target:
                # Find similar recipes with dietary restrictions
                query = f"{base_recipe} {dietary_target}"
                results = await self.rag_system.hybrid_search(
                    query=query,
                    top_k=5,
                    filters={"dietary_restrictions": [dietary_target]}
                )
                
                for result in results:
                    variations.append({
                        "type": "dietary_variation",
                        "dietary_target": dietary_target,
                        "recipe": result,
                        "similarity_to_base": result.get("similarity_score", 0)
                    })
            
            return variations
            
        except Exception as e:
            logger.error(f"Variation generation failed: {e}")
            return []
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server performance statistics."""
        return {
            "server_name": "knowledge-retrieval",
            "search_count": self.search_count,
            "substitution_requests": self.substitution_requests,
            "ingredient_graph_nodes": len(self.ingredient_graph.nodes),
            "technique_count": len(self.technique_db),
            "substitution_mappings": len(self.substitution_db),
            "flavor_profiles": len(self.flavor_profiles)
        }

# Server startup
async def main():
    """Start the Knowledge Retrieval MCP Server."""
    try:
        # Initialize RAG system
        rag_system = EnhancedRAGSystem()
        
        # Initialize server
        server = KnowledgeRetrievalServer(rag_system=rag_system)
        
        # Start server
        logger.info("Starting Knowledge Retrieval MCP Server...")
        await server.server.run()
        
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())