"""
MCP Client for Chef Genius
Orchestrates multiple MCP servers for enhanced recipe generation
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum

import aiohttp
import httpx

logger = logging.getLogger(__name__)

class ServerStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class MCPServerConfig:
    name: str
    url: str
    port: int
    health_endpoint: str = "/health"
    timeout: int = 30

class ChefGeniusMCPClient:
    """
    MCP Client for orchestrating Chef Genius MCP servers.
    
    Handles communication with multiple MCP servers and provides
    high-level recipe generation capabilities.
    """
    
    def __init__(self, server_configs: Optional[List[MCPServerConfig]] = None):
        """
        Initialize the MCP client.
        
        Args:
            server_configs: List of MCP server configurations
        """
        self.server_configs = server_configs or self._get_default_configs()
        self.server_status = {}
        self.performance_metrics = {}
        
        # Connection pools
        self.http_client = None
        
        # Circuit breaker settings
        self.circuit_breakers = {}
        self.max_failures = 3
        self.failure_timeout = 60  # seconds
        
        # Performance tracking
        self.request_count = 0
        self.success_count = 0
        self.error_count = 0
        
        self._initialize_client()
    
    def _get_default_configs(self) -> List[MCPServerConfig]:
        """Get default MCP server configurations."""
        return [
            MCPServerConfig(
                name="recipe-generation",
                url="http://localhost",
                port=8001,
                health_endpoint="/health"
            ),
            MCPServerConfig(
                name="knowledge-retrieval",
                url="http://localhost",
                port=8002,
                health_endpoint="/health"
            ),
            MCPServerConfig(
                name="tool-integration",
                url="http://localhost",
                port=8003,
                health_endpoint="/health"
            )
        ]
    
    def _initialize_client(self):
        """Initialize HTTP client and circuit breakers."""
        # Initialize HTTP client with connection pooling
        timeout = httpx.Timeout(30.0, connect=10.0)
        self.http_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20)
        )
        
        # Initialize circuit breakers
        for config in self.server_configs:
            self.circuit_breakers[config.name] = {
                "failures": 0,
                "last_failure": None,
                "state": "closed"  # closed, open, half-open
            }
            
            self.server_status[config.name] = ServerStatus.UNKNOWN
            self.performance_metrics[config.name] = {
                "request_count": 0,
                "success_count": 0,
                "error_count": 0,
                "avg_response_time": 0,
                "response_times": []
            }
    
    async def __aenter__(self):
        """Async context manager entry."""
        await self.health_check_all_servers()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.http_client:
            await self.http_client.aclose()
    
    async def health_check_all_servers(self) -> Dict[str, ServerStatus]:
        """Check health of all MCP servers."""
        health_tasks = []
        
        for config in self.server_configs:
            task = self._check_server_health(config)
            health_tasks.append(task)
        
        results = await asyncio.gather(*health_tasks, return_exceptions=True)
        
        # Update server status
        for i, result in enumerate(results):
            config = self.server_configs[i]
            if isinstance(result, Exception):
                self.server_status[config.name] = ServerStatus.UNHEALTHY
                logger.error(f"Health check failed for {config.name}: {result}")
            else:
                self.server_status[config.name] = result
        
        return self.server_status
    
    async def _check_server_health(self, config: MCPServerConfig) -> ServerStatus:
        """Check health of a single MCP server."""
        try:
            url = f"{config.url}:{config.port}{config.health_endpoint}"
            response = await self.http_client.get(url, timeout=5.0)
            
            if response.status_code == 200:
                return ServerStatus.HEALTHY
            else:
                return ServerStatus.UNHEALTHY
                
        except Exception as e:
            logger.warning(f"Health check failed for {config.name}: {e}")
            return ServerStatus.UNHEALTHY
    
    def _is_circuit_open(self, server_name: str) -> bool:
        """Check if circuit breaker is open for a server."""
        breaker = self.circuit_breakers.get(server_name, {})
        
        if breaker.get("state") == "open":
            # Check if timeout has passed
            last_failure = breaker.get("last_failure")
            if last_failure and time.time() - last_failure > self.failure_timeout:
                # Move to half-open state
                breaker["state"] = "half-open"
                return False
            return True
        
        return False
    
    def _record_success(self, server_name: str, response_time: float):
        """Record successful request."""
        # Reset circuit breaker
        self.circuit_breakers[server_name] = {
            "failures": 0,
            "last_failure": None,
            "state": "closed"
        }
        
        # Update metrics
        metrics = self.performance_metrics[server_name]
        metrics["request_count"] += 1
        metrics["success_count"] += 1
        metrics["response_times"].append(response_time)
        
        # Keep only last 100 response times
        if len(metrics["response_times"]) > 100:
            metrics["response_times"] = metrics["response_times"][-100:]
        
        # Update average
        metrics["avg_response_time"] = sum(metrics["response_times"]) / len(metrics["response_times"])
        
        self.success_count += 1
    
    def _record_failure(self, server_name: str):
        """Record failed request."""
        breaker = self.circuit_breakers[server_name]
        breaker["failures"] += 1
        breaker["last_failure"] = time.time()
        
        # Open circuit if too many failures
        if breaker["failures"] >= self.max_failures:
            breaker["state"] = "open"
            logger.warning(f"Circuit breaker opened for {server_name}")
        
        # Update metrics
        metrics = self.performance_metrics[server_name]
        metrics["request_count"] += 1
        metrics["error_count"] += 1
        
        self.error_count += 1
    
    async def _call_server_tool(self, server_name: str, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Call a tool on a specific MCP server."""
        self.request_count += 1
        
        # Check circuit breaker
        if self._is_circuit_open(server_name):
            raise Exception(f"Circuit breaker open for {server_name}")
        
        # Find server config
        config = next((c for c in self.server_configs if c.name == server_name), None)
        if not config:
            raise ValueError(f"Unknown server: {server_name}")
        
        start_time = time.time()
        
        try:
            url = f"{config.url}:{config.port}/tools/{tool_name}"
            
            response = await self.http_client.post(
                url,
                json=kwargs,
                timeout=config.timeout
            )
            
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                self._record_success(server_name, response_time)
                return result
            else:
                self._record_failure(server_name)
                raise Exception(f"Server error: {response.status_code} - {response.text}")
                
        except Exception as e:
            self._record_failure(server_name)
            logger.error(f"Tool call failed for {server_name}.{tool_name}: {e}")
            raise
    
    async def generate_enhanced_recipe(
        self,
        ingredients: List[str],
        cuisine: Optional[str] = None,
        dietary_restrictions: Optional[List[str]] = None,
        cooking_time: Optional[int] = None,
        difficulty: Optional[str] = None,
        servings: Optional[int] = None,
        meal_type: Optional[str] = None,
        include_nutrition: bool = True,
        include_substitutions: bool = True,
        include_wine_pairing: bool = False
    ) -> Dict[str, Any]:
        """
        Generate an enhanced recipe using orchestrated MCP servers.
        
        Args:
            ingredients: List of ingredients to include
            cuisine: Cuisine type
            dietary_restrictions: Dietary restrictions
            cooking_time: Maximum cooking time
            difficulty: Difficulty level
            servings: Number of servings
            meal_type: Type of meal
            include_nutrition: Whether to include nutrition analysis
            include_substitutions: Whether to include ingredient substitutions
            include_wine_pairing: Whether to include wine pairing suggestions
            
        Returns:
            Enhanced recipe with all requested information
        """
        try:
            # Step 1: Generate base recipe
            recipe_result = await self._call_server_tool(
                "recipe-generation",
                "generate_recipe",
                ingredients=ingredients,
                cuisine=cuisine,
                dietary_restrictions=dietary_restrictions,
                cooking_time=cooking_time,
                difficulty=difficulty,
                servings=servings,
                meal_type=meal_type
            )
            
            if not recipe_result.get("success", False):
                return {
                    "success": False,
                    "error": "Recipe generation failed",
                    "details": recipe_result
                }
            
            enhanced_result = {
                "recipe": recipe_result["recipe"],
                "generation_metadata": recipe_result.get("generation_metadata", {}),
                "context_recipes": recipe_result.get("context_recipes", []),
                "confidence_score": recipe_result.get("confidence_score", 0.5),
                "success": True
            }
            
            # Step 2: Validate recipe
            try:
                validation_result = await self._call_server_tool(
                    "recipe-generation",
                    "validate_recipe",
                    recipe=recipe_result["recipe"]
                )
                enhanced_result["validation"] = validation_result
            except Exception as e:
                logger.warning(f"Recipe validation failed: {e}")
                enhanced_result["validation"] = {"error": str(e)}
            
            # Step 3: Nutrition analysis (if requested)
            if include_nutrition:
                try:
                    nutrition_result = await self._call_server_tool(
                        "tool-integration",
                        "analyze_nutrition",
                        recipe=recipe_result["recipe"],
                        servings=servings or 4,
                        detailed=True
                    )
                    enhanced_result["nutrition"] = nutrition_result
                except Exception as e:
                    logger.warning(f"Nutrition analysis failed: {e}")
                    enhanced_result["nutrition"] = {"error": str(e)}
            
            # Step 4: Ingredient substitutions (if requested)
            if include_substitutions:
                try:
                    substitution_tasks = []
                    for ingredient in ingredients[:5]:  # Limit to first 5 ingredients
                        task = self._call_server_tool(
                            "knowledge-retrieval",
                            "get_ingredient_substitutions",
                            ingredient=ingredient,
                            dietary_restrictions=dietary_restrictions,
                            recipe_context=recipe_result["recipe"]
                        )
                        substitution_tasks.append((ingredient, task))
                    
                    substitutions = {}
                    for ingredient, task in substitution_tasks:
                        try:
                            result = await task
                            if result:
                                substitutions[ingredient] = result
                        except Exception as e:
                            logger.warning(f"Substitution lookup failed for {ingredient}: {e}")
                    
                    enhanced_result["substitutions"] = substitutions
                except Exception as e:
                    logger.warning(f"Substitution analysis failed: {e}")
                    enhanced_result["substitutions"] = {"error": str(e)}
            
            # Step 5: Wine pairing (if requested)
            if include_wine_pairing:
                try:
                    wine_result = await self._call_server_tool(
                        "tool-integration",
                        "suggest_wine_pairing",
                        recipe=recipe_result["recipe"],
                        occasion="dinner",
                        budget_range="moderate"
                    )
                    enhanced_result["wine_pairing"] = wine_result
                except Exception as e:
                    logger.warning(f"Wine pairing failed: {e}")
                    enhanced_result["wine_pairing"] = {"error": str(e)}
            
            # Step 6: Cooking time estimation
            try:
                time_result = await self._call_server_tool(
                    "tool-integration",
                    "estimate_cooking_time",
                    recipe=recipe_result["recipe"],
                    skill_level="intermediate"
                )
                enhanced_result["time_estimate"] = time_result
            except Exception as e:
                logger.warning(f"Time estimation failed: {e}")
                enhanced_result["time_estimate"] = {"error": str(e)}
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Enhanced recipe generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "fallback_available": True
            }
    
    async def conversational_recipe_generation(
        self,
        conversation_history: List[Dict[str, str]],
        current_request: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle conversational recipe generation with context.
        
        Args:
            conversation_history: Previous conversation messages
            current_request: Current user request
            user_preferences: User preferences and constraints
            
        Returns:
            Conversational response with recipe
        """
        try:
            # Extract requirements from conversation
            requirements = self._extract_requirements_from_conversation(
                conversation_history, current_request, user_preferences
            )
            
            # Generate enhanced recipe
            result = await self.generate_enhanced_recipe(**requirements)
            
            # Add conversational context
            if result.get("success"):
                result["conversation_context"] = {
                    "previous_requests": len(conversation_history),
                    "extracted_requirements": requirements,
                    "conversation_summary": self._summarize_conversation(conversation_history),
                    "personalization_applied": user_preferences is not None
                }
            
            return result
            
        except Exception as e:
            logger.error(f"Conversational generation failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "conversation_error": True
            }
    
    def _extract_requirements_from_conversation(
        self,
        conversation_history: List[Dict[str, str]],
        current_request: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Extract recipe requirements from conversation context."""
        requirements = {}
        
        # Combine all messages for analysis
        all_text = " ".join([msg.get("content", "") for msg in conversation_history])
        all_text += " " + current_request
        all_text = all_text.lower()
        
        # Extract ingredients mentioned
        common_ingredients = [
            "chicken", "beef", "pork", "fish", "salmon", "tuna",
            "rice", "pasta", "noodles", "bread",
            "tomato", "onion", "garlic", "carrot", "potato",
            "cheese", "milk", "butter", "eggs"
        ]
        
        mentioned_ingredients = []
        for ingredient in common_ingredients:
            if ingredient in all_text:
                mentioned_ingredients.append(ingredient)
        
        if mentioned_ingredients:
            requirements["ingredients"] = mentioned_ingredients
        
        # Extract cuisine
        cuisines = ["italian", "chinese", "mexican", "indian", "japanese", "french", "thai"]
        for cuisine in cuisines:
            if cuisine in all_text:
                requirements["cuisine"] = cuisine.title()
                break
        
        # Extract dietary restrictions
        dietary_terms = {
            "vegetarian": ["vegetarian", "veggie"],
            "vegan": ["vegan"],
            "gluten-free": ["gluten free", "gluten-free", "no gluten"],
            "dairy-free": ["dairy free", "dairy-free", "no dairy", "lactose free"],
            "low-carb": ["low carb", "low-carb", "keto"],
            "healthy": ["healthy", "light", "nutritious"]
        }
        
        dietary_restrictions = []
        for restriction, terms in dietary_terms.items():
            if any(term in all_text for term in terms):
                dietary_restrictions.append(restriction)
        
        if dietary_restrictions:
            requirements["dietary_restrictions"] = dietary_restrictions
        
        # Extract time constraints
        time_keywords = ["quick", "fast", "30 minutes", "hour", "slow"]
        if any(keyword in all_text for keyword in time_keywords[:3]):
            requirements["cooking_time"] = 30
        elif "hour" in all_text:
            requirements["cooking_time"] = 60
        
        # Extract difficulty
        if any(word in all_text for word in ["easy", "simple", "beginner"]):
            requirements["difficulty"] = "easy"
        elif any(word in all_text for word in ["hard", "complex", "advanced"]):
            requirements["difficulty"] = "hard"
        
        # Extract meal type
        meal_types = ["breakfast", "lunch", "dinner", "snack", "dessert"]
        for meal_type in meal_types:
            if meal_type in all_text:
                requirements["meal_type"] = meal_type
                break
        
        # Apply user preferences
        if user_preferences:
            if "default_cuisine" in user_preferences and "cuisine" not in requirements:
                requirements["cuisine"] = user_preferences["default_cuisine"]
            
            if "dietary_restrictions" in user_preferences:
                existing_restrictions = requirements.get("dietary_restrictions", [])
                combined_restrictions = list(set(existing_restrictions + user_preferences["dietary_restrictions"]))
                requirements["dietary_restrictions"] = combined_restrictions
            
            if "default_servings" in user_preferences and "servings" not in requirements:
                requirements["servings"] = user_preferences["default_servings"]
        
        return requirements
    
    def _summarize_conversation(self, conversation_history: List[Dict[str, str]]) -> str:
        """Create a summary of the conversation."""
        if not conversation_history:
            return "New conversation"
        
        total_messages = len(conversation_history)
        user_messages = [msg for msg in conversation_history if msg.get("role") == "user"]
        
        return f"Conversation with {total_messages} messages, {len(user_messages)} user requests"
    
    async def search_recipes_advanced(
        self,
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        include_analysis: bool = True,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        Advanced recipe search with analysis.
        
        Args:
            query: Search query
            filters: Search filters
            include_analysis: Whether to include flavor and technique analysis
            top_k: Number of results
            
        Returns:
            Search results with optional analysis
        """
        try:
            # Perform search
            search_result = await self._call_server_tool(
                "knowledge-retrieval",
                "search_recipes",
                query=query,
                **filters or {},
                top_k=top_k
            )
            
            results = {
                "recipes": search_result,
                "search_metadata": {
                    "query": query,
                    "filters": filters,
                    "result_count": len(search_result)
                }
            }
            
            # Add analysis if requested
            if include_analysis and search_result:
                try:
                    # Analyze common ingredients across results
                    all_ingredients = []
                    for recipe in search_result:
                        recipe_ingredients = recipe.get("ingredients", [])
                        all_ingredients.extend(recipe_ingredients)
                    
                    # Get flavor profile analysis
                    if all_ingredients:
                        flavor_analysis = await self._call_server_tool(
                            "knowledge-retrieval",
                            "analyze_flavor_profile",
                            ingredients=list(set(all_ingredients[:10])),  # Unique ingredients, max 10
                            cuisine=filters.get("cuisine") if filters else None
                        )
                        results["flavor_analysis"] = flavor_analysis
                    
                except Exception as e:
                    logger.warning(f"Analysis failed: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Advanced search failed: {e}")
            return {
                "recipes": [],
                "error": str(e)
            }
    
    async def generate_meal_plan(
        self,
        dietary_goals: Dict[str, Any],
        days: int = 7,
        meals_per_day: int = 3,
        preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generate comprehensive meal plan.
        
        Args:
            dietary_goals: Nutritional and dietary goals
            days: Number of days to plan
            meals_per_day: Meals per day
            preferences: User preferences
            
        Returns:
            Complete meal plan with shopping list and prep schedule
        """
        try:
            # Generate meal plan
            meal_plan_result = await self._call_server_tool(
                "tool-integration",
                "plan_meal_prep",
                dietary_goals=dietary_goals,
                days=days,
                meals_per_day=meals_per_day,
                preferences=preferences
            )
            
            return meal_plan_result
            
        except Exception as e:
            logger.error(f"Meal plan generation failed: {e}")
            return {
                "error": str(e),
                "success": False
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        # Refresh server health
        await self.health_check_all_servers()
        
        return {
            "client_stats": {
                "total_requests": self.request_count,
                "successful_requests": self.success_count,
                "failed_requests": self.error_count,
                "success_rate": (self.success_count / self.request_count * 100) if self.request_count > 0 else 0
            },
            "server_status": {name: status.value for name, status in self.server_status.items()},
            "performance_metrics": self.performance_metrics,
            "circuit_breakers": {
                name: {
                    "state": breaker["state"],
                    "failures": breaker["failures"]
                }
                for name, breaker in self.circuit_breakers.items()
            }
        }
    
    async def close(self):
        """Close the MCP client and cleanup resources."""
        if self.http_client:
            await self.http_client.aclose()

# Example usage and testing
async def main():
    """Example usage of the MCP client."""
    async with ChefGeniusMCPClient() as client:
        # Example: Generate enhanced recipe
        result = await client.generate_enhanced_recipe(
            ingredients=["chicken", "rice", "broccoli"],
            cuisine="Asian",
            dietary_restrictions=["healthy"],
            include_nutrition=True,
            include_substitutions=True
        )
        
        print("Enhanced Recipe Result:")
        print(json.dumps(result, indent=2))
        
        # Example: System status
        status = await client.get_system_status()
        print("\nSystem Status:")
        print(json.dumps(status, indent=2))

if __name__ == "__main__":
    asyncio.run(main())