"""
Helper methods for enhanced recipe generator with MCP integration
"""

import logging
import re
from typing import Dict, List, Any, Optional
from app.models.recipe import RecipeCreate, RecipeGenerationRequest

logger = logging.getLogger(__name__)

class RecipeGeneratorHelper:
    """Helper methods for recipe generation and conversion."""
    
    @staticmethod
    def convert_mcp_result_to_recipe(mcp_result: Dict[str, Any], request: RecipeGenerationRequest) -> RecipeCreate:
        """Convert MCP server result to RecipeCreate format."""
        try:
            recipe_text = mcp_result.get("recipe", "")
            
            # Initialize with defaults
            title = "Generated Recipe"
            description = ""
            ingredients = []
            instructions = []
            prep_time = 15
            cook_time = 30
            servings = request.servings or 4
            difficulty = request.difficulty or "medium"
            
            # Parse the recipe text (assuming it's in structured format)
            if isinstance(recipe_text, str):
                parsed_data = RecipeGeneratorHelper._parse_structured_recipe_text(recipe_text)
                
                title = parsed_data.get("title", title)
                description = parsed_data.get("description", description)
                ingredients = parsed_data.get("ingredients", ingredients)
                instructions = parsed_data.get("instructions", instructions)
                prep_time = parsed_data.get("prep_time", prep_time)
                cook_time = parsed_data.get("cook_time", cook_time)
                servings = parsed_data.get("servings", servings)
                difficulty = parsed_data.get("difficulty", difficulty)
            
            elif isinstance(recipe_text, dict):
                # Recipe is already structured
                title = recipe_text.get("title", title)
                description = recipe_text.get("description", description)
                ingredients = recipe_text.get("ingredients", ingredients)
                instructions = recipe_text.get("instructions", instructions)
                prep_time = recipe_text.get("prep_time", prep_time)
                cook_time = recipe_text.get("cook_time", cook_time)
                servings = recipe_text.get("servings", servings)
                difficulty = recipe_text.get("difficulty", difficulty)
            
            # Ensure ingredients are in correct format
            formatted_ingredients = []
            for ingredient in ingredients:
                if isinstance(ingredient, str):
                    formatted_ingredients.append({
                        "name": ingredient,
                        "amount": 1,
                        "unit": "piece",
                        "notes": None
                    })
                elif isinstance(ingredient, dict):
                    formatted_ingredients.append({
                        "name": ingredient.get("name", "unknown ingredient"),
                        "amount": ingredient.get("amount", 1),
                        "unit": ingredient.get("unit", "piece"),
                        "notes": ingredient.get("notes")
                    })
            
            # Ensure instructions are strings
            formatted_instructions = []
            for instruction in instructions:
                if isinstance(instruction, str):
                    formatted_instructions.append(instruction)
                else:
                    formatted_instructions.append(str(instruction))
            
            recipe = RecipeCreate(
                title=title,
                description=description or f"A delicious {request.cuisine or ''} recipe".strip(),
                ingredients=formatted_ingredients,
                instructions=formatted_instructions,
                prep_time=prep_time,
                cook_time=cook_time,
                servings=servings,
                difficulty=difficulty,
                cuisine=request.cuisine,
                dietary_tags=request.dietary_restrictions or []
            )
            
            # Add MCP metadata if available
            if hasattr(recipe, 'metadata'):
                recipe.metadata = {
                    "mcp_enhanced": True,
                    "confidence_score": mcp_result.get("confidence_score", 0.8),
                    "context_recipes": mcp_result.get("context_recipes", []),
                    "generation_metadata": mcp_result.get("generation_metadata", {}),
                    "nutrition": mcp_result.get("nutrition"),
                    "substitutions": mcp_result.get("substitutions"),
                    "validation": mcp_result.get("validation")
                }
            
            return recipe
            
        except Exception as e:
            logger.error(f"Failed to convert MCP result to recipe: {e}")
            return RecipeGeneratorHelper._create_fallback_recipe(request)
    
    @staticmethod
    def _parse_structured_recipe_text(recipe_text: str) -> Dict[str, Any]:
        """Parse structured recipe text into components."""
        try:
            result = {
                "title": "Generated Recipe",
                "description": "",
                "ingredients": [],
                "instructions": [],
                "prep_time": 15,
                "cook_time": 30,
                "servings": 4,
                "difficulty": "medium"
            }
            
            lines = recipe_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Check for section headers
                line_lower = line.lower()
                
                if any(header in line_lower for header in ['title:', '**title:**']):
                    result["title"] = re.sub(r'\*\*title:\*\*|title:', '', line, flags=re.IGNORECASE).strip()
                elif any(header in line_lower for header in ['description:', '**description:**']):
                    result["description"] = re.sub(r'\*\*description:\*\*|description:', '', line, flags=re.IGNORECASE).strip()
                elif any(header in line_lower for header in ['prep time:', '**prep time:**']):
                    time_str = re.sub(r'\*\*prep time:\*\*|prep time:', '', line, flags=re.IGNORECASE).strip()
                    result["prep_time"] = RecipeGeneratorHelper._extract_time_minutes(time_str)
                elif any(header in line_lower for header in ['cook time:', '**cook time:**']):
                    time_str = re.sub(r'\*\*cook time:\*\*|cook time:', '', line, flags=re.IGNORECASE).strip()
                    result["cook_time"] = RecipeGeneratorHelper._extract_time_minutes(time_str)
                elif any(header in line_lower for header in ['servings:', '**servings:**']):
                    serv_str = re.sub(r'\*\*servings:\*\*|servings:', '', line, flags=re.IGNORECASE).strip()
                    result["servings"] = RecipeGeneratorHelper._extract_servings(serv_str)
                elif any(header in line_lower for header in ['difficulty:', '**difficulty:**']):
                    result["difficulty"] = re.sub(r'\*\*difficulty:\*\*|difficulty:', '', line, flags=re.IGNORECASE).strip().lower()
                elif any(header in line_lower for header in ['ingredients:', '**ingredients:**']):
                    current_section = "ingredients"
                    continue
                elif any(header in line_lower for header in ['instructions:', '**instructions:**', 'method:', '**method:**']):
                    current_section = "instructions"
                    continue
                
                # Parse content based on current section
                if current_section == "ingredients":
                    ingredient = RecipeGeneratorHelper._parse_ingredient_line(line)
                    if ingredient:
                        result["ingredients"].append(ingredient)
                elif current_section == "instructions":
                    # Clean up instruction line
                    instruction = re.sub(r'^\d+\.\s*', '', line)  # Remove leading numbers
                    if instruction and not instruction.startswith('**'):
                        result["instructions"].append(instruction)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to parse structured recipe text: {e}")
            return {
                "title": "Generated Recipe",
                "description": "A delicious recipe",
                "ingredients": [],
                "instructions": ["Follow basic cooking steps"],
                "prep_time": 15,
                "cook_time": 30,
                "servings": 4,
                "difficulty": "medium"
            }
    
    @staticmethod
    def _extract_time_minutes(time_str: str) -> int:
        """Extract minutes from time string."""
        time_str = time_str.lower()
        if "hour" in time_str:
            hours = re.search(r'(\d+)\s*hour', time_str)
            minutes = re.search(r'(\d+)\s*minute', time_str)
            total = 0
            if hours:
                total += int(hours.group(1)) * 60
            if minutes:
                total += int(minutes.group(1))
            return total or 30
        else:
            minutes = re.search(r'(\d+)', time_str)
            return int(minutes.group(1)) if minutes else 30
    
    @staticmethod
    def _extract_servings(serv_str: str) -> int:
        """Extract serving count from string."""
        numbers = re.findall(r'\d+', serv_str)
        return int(numbers[0]) if numbers else 4
    
    @staticmethod
    def _parse_ingredient_line(line: str) -> Optional[Dict[str, Any]]:
        """Parse ingredient line into structured format."""
        try:
            # Remove bullet points and dashes
            line = re.sub(r'^[-*•]\s*', '', line.strip())
            
            # Pattern to match "amount unit ingredient" format
            pattern = re.compile(r'^(\d+(?:\.\d+)?(?:/\d+)?)\s*(\w+)?\s+(.+)$')
            match = pattern.match(line)
            
            if match:
                amount_str, unit, name = match.groups()
                # Convert fractions
                if '/' in amount_str:
                    parts = amount_str.split('/')
                    amount = float(parts[0]) / float(parts[1])
                else:
                    amount = float(amount_str)
                
                return {
                    "name": name.strip(),
                    "amount": amount,
                    "unit": unit or "piece",
                    "notes": None
                }
            else:
                # Simple ingredient without amount
                return {
                    "name": line.strip(),
                    "amount": 1,
                    "unit": "piece",
                    "notes": None
                }
        except Exception:
            return {
                "name": line.strip(),
                "amount": 1,
                "unit": "piece",
                "notes": None
            }
    
    @staticmethod
    def adapt_recipe_from_rag(base_recipe: Dict[str, Any], request: RecipeGenerationRequest) -> RecipeCreate:
        """Adapt a RAG-found recipe to match user requirements."""
        try:
            # Extract base recipe data
            title = base_recipe.get("title", "Adapted Recipe")
            base_ingredients = base_recipe.get("ingredients", [])
            base_instructions = base_recipe.get("instructions", [])
            cuisine = base_recipe.get("cuisine", request.cuisine)
            
            # Adapt ingredients based on user requirements
            adapted_ingredients = []
            user_ingredients = set((request.ingredients or []))
            
            # Include user-specified ingredients
            for user_ingredient in user_ingredients:
                adapted_ingredients.append({
                    "name": user_ingredient,
                    "amount": 1,
                    "unit": "cup",
                    "notes": None
                })
            \n            # Add compatible ingredients from base recipe\n            for ingredient in base_ingredients:\n                ingredient_name = ingredient if isinstance(ingredient, str) else ingredient.get(\"name\", \"\")\n                \n                # Check if ingredient is compatible with dietary restrictions\n                if RecipeGeneratorHelper._is_ingredient_compatible(ingredient_name, request.dietary_restrictions):\n                    if isinstance(ingredient, str):\n                        adapted_ingredients.append({\n                            \"name\": ingredient,\n                            \"amount\": 1,\n                            \"unit\": \"piece\",\n                            \"notes\": None\n                        })\n                    else:\n                        adapted_ingredients.append(ingredient)\n            \n            # Adapt instructions\n            adapted_instructions = []\n            if isinstance(base_instructions, list):\n                for instruction in base_instructions:\n                    if isinstance(instruction, str):\n                        adapted_instructions.append(instruction)\n                    else:\n                        adapted_instructions.append(str(instruction))\n            else:\n                adapted_instructions = [str(base_instructions)]\n            \n            # Create adapted recipe\n            recipe = RecipeCreate(\n                title=f\"Adapted {title}\",\n                description=f\"A {cuisine or 'delicious'} recipe adapted for your preferences\",\n                ingredients=adapted_ingredients,\n                instructions=adapted_instructions,\n                prep_time=base_recipe.get(\"prep_time\", 15),\n                cook_time=base_recipe.get(\"cook_time\", 30),\n                servings=request.servings or base_recipe.get(\"servings\", 4),\n                difficulty=request.difficulty or base_recipe.get(\"difficulty\", \"medium\"),\n                cuisine=cuisine,\n                dietary_tags=request.dietary_restrictions or []\n            )\n            \n            # Add RAG metadata\n            if hasattr(recipe, 'metadata'):\n                recipe.metadata = {\n                    \"rag_adapted\": True,\n                    \"base_recipe_similarity\": base_recipe.get(\"similarity_score\", 0.7),\n                    \"adaptation_method\": \"ingredient_substitution_and_instruction_adaptation\"\n                }\n            \n            return recipe\n            \n        except Exception as e:\n            logger.error(f\"Failed to adapt RAG recipe: {e}\")\n            return RecipeGeneratorHelper._create_fallback_recipe(request)\n    \n    @staticmethod\n    def _is_ingredient_compatible(ingredient_name: str, dietary_restrictions: Optional[List[str]]) -> bool:\n        \"\"\"Check if ingredient is compatible with dietary restrictions.\"\"\"\n        if not dietary_restrictions:\n            return True\n        \n        ingredient_lower = ingredient_name.lower()\n        \n        # Common dietary restriction checks\n        if \"vegan\" in dietary_restrictions:\n            non_vegan_items = [\"meat\", \"chicken\", \"beef\", \"pork\", \"fish\", \"milk\", \"cheese\", \"butter\", \"egg\"]\n            if any(item in ingredient_lower for item in non_vegan_items):\n                return False\n        \n        if \"vegetarian\" in dietary_restrictions:\n            non_vegetarian_items = [\"meat\", \"chicken\", \"beef\", \"pork\", \"fish\", \"bacon\"]\n            if any(item in ingredient_lower for item in non_vegetarian_items):\n                return False\n        \n        if \"gluten-free\" in dietary_restrictions or \"gluten free\" in dietary_restrictions:\n            gluten_items = [\"wheat\", \"flour\", \"bread\", \"pasta\", \"barley\", \"rye\"]\n            if any(item in ingredient_lower for item in gluten_items):\n                return False\n        \n        if \"dairy-free\" in dietary_restrictions or \"dairy free\" in dietary_restrictions:\n            dairy_items = [\"milk\", \"cheese\", \"butter\", \"cream\", \"yogurt\"]\n            if any(item in ingredient_lower for item in dairy_items):\n                return False\n        \n        return True\n    \n    @staticmethod\n    def _create_fallback_recipe(request: RecipeGenerationRequest) -> RecipeCreate:\n        \"\"\"Create a simple fallback recipe if all other methods fail.\"\"\"\n        ingredients_str = \", \".join(request.ingredients) if request.ingredients else \"mixed vegetables\"\n        \n        return RecipeCreate(\n            title=f\"Simple {request.cuisine or 'Home'} Style {ingredients_str.title()}\",\n            description=\"A simple and delicious recipe created for you.\",\n            ingredients=[\n                {\"name\": ingredient, \"amount\": 1, \"unit\": \"cup\", \"notes\": None}\n                for ingredient in (request.ingredients or [\"vegetables\", \"olive oil\", \"salt\"])\n            ],\n            instructions=[\n                \"Prepare all ingredients as listed.\",\n                \"Cook ingredients using your preferred method.\",\n                \"Season to taste with salt and pepper.\",\n                \"Serve hot and enjoy!\"\n            ],\n            prep_time=10,\n            cook_time=20,\n            servings=request.servings or 4,\n            difficulty=request.difficulty or \"easy\",\n            cuisine=request.cuisine,\n            dietary_tags=request.dietary_restrictions or []\n        )