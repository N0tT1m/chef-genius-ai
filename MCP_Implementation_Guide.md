# MCP Implementation Guide for Chef Genius

## MCP Server Architecture

### Core MCP Servers

#### 1. Recipe Generation Server
```python
# /Users/timmy/workspace/ai-apps/chef-genius/mcp_servers/recipe_server.py

from mcp import Server, Tool
from mcp.types import TextContent, EmbeddedResource
import asyncio
from typing import List, Optional, Dict, Any

class RecipeGenerationServer:
    def __init__(self):
        self.server = Server("recipe-generation")
        self.t5_model = None  # Your fine-tuned T5-Large
        self.rag_system = None  # Your existing RAG system
        
    @self.server.tool("generate_recipe")
    async def generate_recipe(
        ingredients: List[str],
        cuisine: Optional[str] = None,
        dietary_restrictions: Optional[List[str]] = None,
        cooking_time: Optional[int] = None,
        difficulty: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a recipe using T5-Large + RAG context"""
        
        # 1. RAG retrieval for context
        search_query = " ".join(ingredients)
        if cuisine:
            search_query += f" {cuisine}"
            
        similar_recipes = await self.rag_system.search_similar_recipes(
            query=search_query,
            top_k=3,
            min_similarity=0.4
        )
        
        # 2. Create enhanced prompt
        prompt = self._create_enhanced_prompt(
            ingredients=ingredients,
            cuisine=cuisine,
            dietary_restrictions=dietary_restrictions,
            cooking_time=cooking_time,
            difficulty=difficulty,
            context_recipes=similar_recipes
        )
        
        # 3. Generate with T5-Large
        generated_recipe = await self._generate_with_t5(prompt)
        
        # 4. Post-process and validate
        validated_recipe = await self._validate_recipe(generated_recipe)
        
        return {
            "recipe": validated_recipe,
            "context_recipes": [r["title"] for r in similar_recipes],
            "confidence_score": self._calculate_confidence(validated_recipe),
            "generation_metadata": {
                "model": "t5-large-recipes",
                "rag_enhanced": True,
                "context_count": len(similar_recipes)
            }
        }
    
    @self.server.tool("refine_recipe")
    async def refine_recipe(
        original_recipe: str,
        refinement_request: str
    ) -> Dict[str, Any]:
        """Refine an existing recipe based on user feedback"""
        
        # Use RAG to find similar refinement patterns
        refinement_context = await self.rag_system.search_similar_recipes(
            query=f"{original_recipe} {refinement_request}",
            top_k=2
        )
        
        # Generate refined version
        refined_recipe = await self._refine_with_context(
            original_recipe, 
            refinement_request, 
            refinement_context
        )
        
        return {
            "refined_recipe": refined_recipe,
            "changes_made": self._identify_changes(original_recipe, refined_recipe),
            "refinement_confidence": self._calculate_refinement_confidence(refined_recipe)
        }
    
    @self.server.tool("validate_recipe")
    async def validate_recipe(recipe: str) -> Dict[str, Any]:
        """Validate recipe structure and nutritional balance"""
        
        validation_results = {
            "structure_valid": self._check_recipe_structure(recipe),
            "ingredients_valid": self._validate_ingredients(recipe),
            "instructions_clear": self._validate_instructions(recipe),
            "nutritional_balance": await self._check_nutritional_balance(recipe),
            "cooking_time_realistic": self._validate_cooking_time(recipe),
            "difficulty_appropriate": self._validate_difficulty(recipe)
        }
        
        overall_score = sum(validation_results.values()) / len(validation_results)
        
        return {
            "validation_results": validation_results,
            "overall_score": overall_score,
            "suggestions": self._generate_improvement_suggestions(validation_results),
            "valid": overall_score >= 0.8
        }
```

#### 2. Knowledge Retrieval Server
```python
# /Users/timmy/workspace/ai-apps/chef-genius/mcp_servers/knowledge_server.py

class KnowledgeRetrievalServer:
    def __init__(self):
        self.server = Server("knowledge-retrieval")
        self.rag_system = None
        self.knowledge_graph = None
        
    @self.server.tool("search_recipes")
    async def search_recipes(
        query: str,
        filters: Optional[Dict[str, Any]] = None,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Search recipe database with advanced filtering"""
        
        # Hybrid search: semantic + keyword + filters
        results = await self.rag_system.search_similar_recipes(
            query=query,
            top_k=top_k * 2  # Get more for filtering
        )
        
        # Apply filters
        if filters:
            results = self._apply_filters(results, filters)
        
        return results[:top_k]
    
    @self.server.tool("get_ingredient_substitutions")
    async def get_ingredient_substitutions(
        ingredient: str,
        dietary_restrictions: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Get substitution suggestions for ingredients"""
        
        # Query knowledge graph for substitutions
        substitutions = await self.knowledge_graph.get_substitutions(
            ingredient=ingredient,
            restrictions=dietary_restrictions
        )
        
        # Enhance with RAG context
        for sub in substitutions:
            examples = await self.rag_system.search_similar_recipes(
                query=f"{ingredient} substitute {sub['substitute']}",
                top_k=2
            )
            sub["usage_examples"] = examples
        
        return substitutions
    
    @self.server.tool("get_cooking_techniques")
    async def get_cooking_techniques(
        ingredient: str,
        cuisine: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get cooking techniques for specific ingredients"""
        
        query = f"cooking techniques {ingredient}"
        if cuisine:
            query += f" {cuisine}"
            
        technique_recipes = await self.rag_system.search_similar_recipes(
            query=query,
            top_k=10
        )
        
        # Extract and categorize techniques
        techniques = self._extract_techniques(technique_recipes)
        
        return techniques
    
    @self.server.tool("get_recipe_variations")
    async def get_recipe_variations(
        base_recipe: str,
        variation_type: str = "cuisine"  # cuisine, dietary, difficulty
    ) -> List[Dict[str, Any]]:
        """Get variations of a base recipe"""
        
        variations = await self.rag_system.search_similar_recipes(
            query=f"{base_recipe} {variation_type} variation",
            top_k=5
        )
        
        return variations
```

#### 3. Tool Integration Server
```python
# /Users/timmy/workspace/ai-apps/chef-genius/mcp_servers/tool_server.py

class ToolIntegrationServer:
    def __init__(self):
        self.server = Server("tool-integration")
        self.nutrition_service = None
        self.substitution_service = None
        self.vision_service = None
        
    @self.server.tool("analyze_nutrition")
    async def analyze_nutrition(recipe: str) -> Dict[str, Any]:
        """Analyze nutritional content of a recipe"""
        return await self.nutrition_service.analyze_recipe(recipe)
    
    @self.server.tool("generate_shopping_list")
    async def generate_shopping_list(
        recipes: List[str],
        servings: List[int]
    ) -> Dict[str, Any]:
        """Generate optimized shopping list for multiple recipes"""
        
        # Extract ingredients from all recipes
        all_ingredients = []
        for i, recipe in enumerate(recipes):
            ingredients = self._extract_ingredients(recipe)
            scaled_ingredients = self._scale_ingredients(ingredients, servings[i])
            all_ingredients.extend(scaled_ingredients)
        
        # Consolidate and optimize
        consolidated = self._consolidate_ingredients(all_ingredients)
        optimized = self._optimize_shopping_list(consolidated)
        
        return {
            "shopping_list": optimized,
            "estimated_cost": self._estimate_cost(optimized),
            "store_suggestions": self._suggest_stores(optimized)
        }
    
    @self.server.tool("analyze_food_image")
    async def analyze_food_image(image_url: str) -> Dict[str, Any]:
        """Analyze food image to identify ingredients and suggest recipes"""
        
        # Use vision service to identify ingredients
        identified_ingredients = await self.vision_service.identify_ingredients(image_url)
        
        # Find recipes using identified ingredients
        recipe_suggestions = await self.rag_system.search_similar_recipes(
            query=" ".join(identified_ingredients),
            top_k=3
        )
        
        return {
            "identified_ingredients": identified_ingredients,
            "recipe_suggestions": recipe_suggestions,
            "confidence_scores": self._calculate_vision_confidence(identified_ingredients)
        }
    
    @self.server.tool("plan_meal_prep")
    async def plan_meal_prep(
        dietary_goals: Dict[str, Any],
        days: int = 7,
        meals_per_day: int = 3
    ) -> Dict[str, Any]:
        """Generate meal prep plan based on dietary goals"""
        
        # Generate meal plan using RAG
        meal_plan = await self._generate_meal_plan(dietary_goals, days, meals_per_day)
        
        # Optimize for prep efficiency
        prep_schedule = self._optimize_prep_schedule(meal_plan)
        shopping_list = await self.generate_shopping_list(
            [meal["recipe"] for meal in meal_plan],
            [meal["servings"] for meal in meal_plan]
        )
        
        return {
            "meal_plan": meal_plan,
            "prep_schedule": prep_schedule,
            "shopping_list": shopping_list,
            "nutritional_summary": self._summarize_nutrition(meal_plan)
        }
```

## MCP Client Integration

### Client-Side Integration
```python
# /Users/timmy/workspace/ai-apps/chef-genius/backend/app/services/mcp_client.py

from mcp import Client
import asyncio
from typing import Dict, Any, List, Optional

class ChefGeniusMCPClient:
    def __init__(self):
        self.recipe_client = Client("recipe-generation")
        self.knowledge_client = Client("knowledge-retrieval") 
        self.tool_client = Client("tool-integration")
        
    async def generate_enhanced_recipe(
        self,
        ingredients: List[str],
        cuisine: Optional[str] = None,
        dietary_restrictions: Optional[List[str]] = None,
        cooking_time: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate recipe using MCP orchestration"""
        
        # 1. Generate base recipe
        recipe_result = await self.recipe_client.call_tool(
            "generate_recipe",
            ingredients=ingredients,
            cuisine=cuisine,
            dietary_restrictions=dietary_restrictions,
            cooking_time=cooking_time
        )
        
        # 2. Validate recipe
        validation_result = await self.recipe_client.call_tool(
            "validate_recipe",
            recipe=recipe_result["recipe"]
        )
        
        # 3. Analyze nutrition
        nutrition_result = await self.tool_client.call_tool(
            "analyze_nutrition",
            recipe=recipe_result["recipe"]
        )
        
        # 4. Get ingredient substitutions
        substitution_tasks = []
        for ingredient in ingredients:
            substitution_tasks.append(
                self.knowledge_client.call_tool(
                    "get_ingredient_substitutions",
                    ingredient=ingredient,
                    dietary_restrictions=dietary_restrictions
                )
            )
        
        substitutions = await asyncio.gather(*substitution_tasks)
        
        return {
            "recipe": recipe_result["recipe"],
            "validation": validation_result,
            "nutrition": nutrition_result,
            "substitutions": dict(zip(ingredients, substitutions)),
            "context_recipes": recipe_result["context_recipes"],
            "confidence_score": recipe_result["confidence_score"]
        }
    
    async def conversational_recipe_generation(
        self,
        conversation_history: List[Dict[str, str]],
        current_request: str
    ) -> Dict[str, Any]:
        """Handle conversational recipe generation with context"""
        
        # Extract recipe requirements from conversation
        requirements = self._extract_requirements_from_conversation(
            conversation_history, current_request
        )
        
        # Generate recipe with full context
        result = await self.generate_enhanced_recipe(**requirements)
        
        # Add conversational context
        result["conversation_context"] = {
            "previous_requests": len(conversation_history),
            "refined_requirements": requirements,
            "conversation_summary": self._summarize_conversation(conversation_history)
        }
        
        return result
```

## Integration with Existing Backend

### Modified Recipe Generator Service
```python
# /Users/timmy/workspace/ai-apps/chef-genius/backend/app/services/recipe_generator.py

class RecipeGenerator:
    def __init__(self):
        self.mcp_client = ChefGeniusMCPClient()
        self.t5_model = None  # Your existing T5 model
        self.rag_system = None  # Your existing RAG system
        
    async def generate_recipe(self, request: RecipeRequest) -> RecipeResponse:
        """Enhanced recipe generation with MCP orchestration"""
        
        # Use MCP for enhanced generation
        mcp_result = await self.mcp_client.generate_enhanced_recipe(
            ingredients=request.ingredients,
            cuisine=request.cuisine,
            dietary_restrictions=request.dietary_restrictions,
            cooking_time=request.cooking_time
        )
        
        # Fallback to direct T5 if MCP fails
        if not mcp_result or mcp_result.get("confidence_score", 0) < 0.6:
            fallback_result = await self._generate_with_t5_direct(request)
            return RecipeResponse(
                recipe=fallback_result,
                source="t5-direct",
                confidence=0.5
            )
        
        return RecipeResponse(
            recipe=mcp_result["recipe"],
            nutrition=mcp_result["nutrition"],
            substitutions=mcp_result["substitutions"],
            validation=mcp_result["validation"],
            context_recipes=mcp_result["context_recipes"],
            source="mcp-enhanced",
            confidence=mcp_result["confidence_score"]
        )
```

## Deployment Architecture

### Docker Compose Configuration
```yaml
# /Users/timmy/workspace/ai-apps/chef-genius/docker-compose.mcp.yml

version: '3.8'

services:
  # MCP Servers
  recipe-server:
    build: ./mcp_servers/recipe_server
    ports:
      - "8001:8001"
    environment:
      - MODEL_PATH=/models/t5-large-recipes
      - RAG_DB_PATH=/data/recipes.db
    volumes:
      - ./models:/models
      - ./data:/data
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  
  knowledge-server:
    build: ./mcp_servers/knowledge_server
    ports:
      - "8002:8002"
    environment:
      - VECTOR_DB_URL=http://weaviate:8080
      - RAG_DB_PATH=/data/recipes.db
    volumes:
      - ./data:/data
    depends_on:
      - weaviate
  
  tool-server:
    build: ./mcp_servers/tool_server
    ports:
      - "8003:8003"
    environment:
      - NUTRITION_API_KEY=${NUTRITION_API_KEY}
      - VISION_MODEL_PATH=/models/vision
    volumes:
      - ./models:/models
  
  # Vector Database
  weaviate:
    image: semitechnologies/weaviate:1.21.0
    ports:
      - "8080:8080"
    environment:
      - QUERY_DEFAULTS_LIMIT=25
      - AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true
      - PERSISTENCE_DATA_PATH=/var/lib/weaviate
      - DEFAULT_VECTORIZER_MODULE=none
      - CLUSTER_HOSTNAME=node1
    volumes:
      - weaviate_data:/var/lib/weaviate
  
  # Enhanced Backend
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    environment:
      - MCP_RECIPE_SERVER_URL=http://recipe-server:8001
      - MCP_KNOWLEDGE_SERVER_URL=http://knowledge-server:8002
      - MCP_TOOL_SERVER_URL=http://tool-server:8003
    depends_on:
      - recipe-server
      - knowledge-server
      - tool-server

volumes:
  weaviate_data:
```

## Performance Monitoring

### MCP Server Metrics
```python
# /Users/timmy/workspace/ai-apps/chef-genius/mcp_servers/monitoring.py

import time
import asyncio
from functools import wraps
from prometheus_client import Counter, Histogram, Gauge

# Metrics
REQUEST_COUNT = Counter('mcp_requests_total', 'Total MCP requests', ['server', 'tool'])
REQUEST_DURATION = Histogram('mcp_request_duration_seconds', 'Request duration', ['server', 'tool'])
ACTIVE_CONNECTIONS = Gauge('mcp_active_connections', 'Active connections', ['server'])

def monitor_tool(server_name: str):
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            tool_name = func.__name__
            start_time = time.time()
            
            REQUEST_COUNT.labels(server=server_name, tool=tool_name).inc()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = time.time() - start_time
                REQUEST_DURATION.labels(server=server_name, tool=tool_name).observe(duration)
        
        return wrapper
    return decorator
```

This MCP architecture will provide you with:

1. **Modular Design**: Separate servers for different capabilities
2. **Scalability**: Individual servers can be scaled independently
3. **Reliability**: Fallback mechanisms to your existing T5 model
4. **Monitoring**: Built-in performance tracking
5. **Flexibility**: Easy to add new tools and capabilities

The system leverages your existing 4.1M recipe dataset and T5-Large model while adding powerful orchestration capabilities through MCP.