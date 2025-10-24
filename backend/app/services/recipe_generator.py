import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
import json
import re
from app.models.recipe import RecipeGenerationRequest, RecipeCreate, NutritionInfo
from app.core.config import settings
from app.services.enhanced_rag_system import EnhancedRAGSystem
from app.services.mcp_client import ChefGeniusMCPClient
from app.services.recipe_generator_helper import RecipeGeneratorHelper
from app.services.tool_system import ToolSystem, ToolCall
from app.services.performance_optimizer import PerformanceOptimizer, performance_monitor
import logging

logger = logging.getLogger(__name__)

class RecipeGeneratorService:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # MCP Client for orchestrated generation
        self.mcp_client = None
        
        # Enhanced RAG system for knowledge-enhanced generation
        self.rag_system = None
        
        # Initialize tool system for function calling
        self.tool_system = ToolSystem()
        
        # Initialize performance optimizer
        self.performance_optimizer = PerformanceOptimizer()
        
        # Fallback model components - Updated for Mistral
        self.model_name = settings.RECIPE_MODEL_PATH
        self.tokenizer = None
        self.model = None
        self.generator = None
        
        self._initialize_mcp_client()
        self._initialize_rag_system()
        self._load_fallback_model()
    
    def _initialize_mcp_client(self):
        """Initialize the MCP client for orchestrated generation."""
        try:
            logger.info("Initializing MCP client...")
            self.mcp_client = ChefGeniusMCPClient()
            logger.info("MCP client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize MCP client: {e}")
            self.mcp_client = None
    
    def _initialize_rag_system(self):
        """Initialize the enhanced RAG system for knowledge retrieval."""
        try:
            logger.info("Initializing enhanced RAG system...")
            self.rag_system = EnhancedRAGSystem(
                weaviate_url="http://localhost:8080",
                embedding_model="sentence-transformers/all-MiniLM-L12-v2",
                recipe_db_path="/Users/timmy/workspace/ai-apps/chef-genius/cli/data/training.json"
            )
            logger.info("Enhanced RAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize enhanced RAG system: {e}")
            self.rag_system = None
    
    async def _load_fallback_model(self):
        """Load Mistral-7B-Instruct with 4-bit quantization for RTX 4090."""
        try:
            # Setup 4-bit quantization for Mistral
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=settings.USE_4BIT_QUANTIZATION,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                padding_side="left"
            )
            
            # Load model with quantization
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                quantization_config=bnb_config if settings.USE_4BIT_QUANTIZATION else None,
                torch_dtype=getattr(torch, settings.TORCH_DTYPE) if hasattr(torch, settings.TORCH_DTYPE) else torch.bfloat16,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
                
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                torch_dtype=getattr(torch, settings.TORCH_DTYPE) if hasattr(torch, settings.TORCH_DTYPE) else torch.bfloat16
            )
            
            # Log memory usage
            param_count = sum(p.numel() for p in self.model.parameters())
            logger.info(f"Mistral recipe generation model loaded successfully: {self.model_name}")
            logger.info(f"Model parameters: {param_count:,}")
            logger.info(f"Quantization: {settings.USE_4BIT_QUANTIZATION}")
            
        except Exception as e:
            logger.error(f"Failed to load Mistral model: {e}")
            # Fallback to smaller model
            try:
                self.generator = pipeline(
                    "text-generation", 
                    model="microsoft/DialoGPT-medium",
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("Loaded fallback model")
            except Exception as fallback_error:
                logger.error(f"Fallback model also failed: {fallback_error}")
                self.generator = pipeline("text-generation", model="gpt2")
    
    @performance_monitor("recipe_generation")
    async def generate_recipe(self, request: RecipeGenerationRequest) -> RecipeCreate:
        """Generate a recipe using MCP orchestration with fallbacks."""
        try:
            # Check cache first
            cache_key = {
                "ingredients": request.ingredients,
                "cuisine": request.cuisine,
                "dietary_restrictions": request.dietary_restrictions,
                "difficulty": request.difficulty,
                "meal_type": request.meal_type
            }
            
            cached_recipe = self.performance_optimizer.cache.get(cache_key)
            if cached_recipe:
                logger.info("Returning cached recipe")
                return cached_recipe
            
            # Auto cleanup if needed
            await self.performance_optimizer.auto_cleanup_if_needed()
            
            # Try MCP-enhanced generation first
            if self.mcp_client:
                logger.info("Using MCP-enhanced recipe generation")
                try:
                    async with self.mcp_client as client:
                        mcp_result = await client.generate_enhanced_recipe(
                            ingredients=request.ingredients or [],
                            cuisine=request.cuisine,
                            dietary_restrictions=request.dietary_restrictions,
                            cooking_time=getattr(request, 'cooking_time', None),
                            difficulty=request.difficulty,
                            servings=request.servings,
                            meal_type=request.meal_type,
                            include_nutrition=True,
                            include_substitutions=True
                        )
                        
                        if mcp_result.get("success") and mcp_result.get("recipe"):
                            # Convert MCP result to RecipeCreate format
                            recipe = RecipeGeneratorHelper.convert_mcp_result_to_recipe(mcp_result, request)
                            
                            # Cache the result
                            self.performance_optimizer.cache.put(cache_key, recipe, ttl=1800)
                            
                            logger.info("Successfully generated MCP-enhanced recipe")
                            return recipe
                        else:
                            logger.warning(f"MCP generation failed: {mcp_result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    logger.error(f"MCP generation failed: {e}")
            
            # Try enhanced RAG system as fallback
            if self.rag_system:
                logger.info("Using enhanced RAG-based recipe generation")
                try:
                    # Create search query
                    search_query = " ".join(request.ingredients or ["recipe"])
                    if request.cuisine:
                        search_query += f" {request.cuisine}"
                    
                    # Search for similar recipes
                    similar_recipes = await self.rag_system.hybrid_search(
                        query=search_query,
                        top_k=3,
                        filters={
                            "cuisine": request.cuisine,
                            "dietary_restrictions": request.dietary_restrictions
                        } if request.cuisine or request.dietary_restrictions else None
                    )
                    
                    if similar_recipes:
                        # Use best matching recipe as base and adapt it
                        base_recipe = similar_recipes[0]
                        adapted_recipe = RecipeGeneratorHelper.adapt_recipe_from_rag(base_recipe, request)
                        
                        # Cache the result
                        self.performance_optimizer.cache.put(cache_key, adapted_recipe, ttl=1800)
                        
                        logger.info("Successfully generated RAG-enhanced recipe")
                        return adapted_recipe
                
                except Exception as e:
                    logger.error(f"Enhanced RAG generation failed: {e}")
            
            # Final fallback to direct generation
            logger.info("Using fallback recipe generation")
            if self.generator:
                prompt = self._create_prompt(request)
                generated_text = self._generate_text(prompt)
                recipe = self._parse_recipe(generated_text, request)
            else:
                recipe = self._create_fallback_recipe(request)
            
            # Cache the result
            self.performance_optimizer.cache.put(cache_key, recipe, ttl=1800)
            
            return recipe
            
        except Exception as e:
            logger.error(f"All recipe generation methods failed: {e}")
            # Return a fallback recipe
            return self._create_fallback_recipe(request)
    
    def _create_prompt(self, request: RecipeGenerationRequest) -> str:
        """Create a structured prompt optimized for Mistral's chat template."""
        
        requirements = []
        if request.ingredients:
            ingredients_str = ", ".join(request.ingredients)
            requirements.append(f"Must include these ingredients: {ingredients_str}")
        
        if request.cuisine:
            requirements.append(f"Cuisine style: {request.cuisine}")
        
        if request.dietary_restrictions:
            restrictions = ", ".join(request.dietary_restrictions)
            requirements.append(f"Dietary requirements: {restrictions}")
        
        if request.cooking_time:
            requirements.append(f"Total cooking time: {request.cooking_time}")
        
        if request.difficulty:
            requirements.append(f"Difficulty level: {request.difficulty}")
        
        if request.servings:
            requirements.append(f"Servings: {request.servings}")
        
        if request.meal_type:
            requirements.append(f"Meal type: {request.meal_type}")
        
        user_request = "Create a complete recipe with the following requirements:\n" + "\n".join(f"- {req}" for req in requirements)
        user_request += "\n\nProvide the recipe in this exact format:\n"
        user_request += "**TITLE:** [Recipe Name]\n"
        user_request += "**DESCRIPTION:** [Brief description]\n"
        user_request += "**PREP TIME:** [X minutes]\n"
        user_request += "**COOK TIME:** [X minutes]\n"
        user_request += "**SERVINGS:** [X people]\n"
        user_request += "**DIFFICULTY:** [easy/medium/hard]\n"
        user_request += "**INGREDIENTS:**\n[List ingredients with amounts]\n"
        user_request += "**INSTRUCTIONS:**\n[Step-by-step numbered instructions]"
        
        # Format for Mistral chat template
        messages = [
            {"role": "user", "content": user_request}
        ]
        
        # Apply Mistral's chat template
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        except Exception as e:
            logger.warning(f"Failed to apply chat template: {e}")
            # Fallback to simple format
            prompt = f"<s>[INST] {user_request} [/INST]"
        
        return prompt
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using Mistral with optimized parameters."""
        try:
            # Optimized generation parameters for Mistral
            result = self.generator(
                prompt,
                max_new_tokens=settings.MAX_RECIPE_GENERATION_LENGTH,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                repetition_penalty=1.1,
                length_penalty=1.0,
                pad_token_id=self.generator.tokenizer.pad_token_id,
                eos_token_id=self.generator.tokenizer.eos_token_id,
                return_full_text=False  # Only return generated text, not prompt
            )
            
            # Extract only the generated text
            generated_text = result[0]['generated_text']
            if not generated_text.strip():
                # Fallback if no text generated
                generated_text = self._create_simple_recipe_text(prompt)
                
            return generated_text
        except Exception as e:
            logger.error(f"Text generation failed: {e}")
            return self._create_simple_recipe_text(prompt)
    
    def _create_simple_recipe_text(self, prompt: str) -> str:
        """Create a simple recipe text as fallback."""
        return """**TITLE:** Simple Delicious Recipe
**DESCRIPTION:** A quick and easy recipe for any occasion
**PREP TIME:** 15 minutes
**COOK TIME:** 25 minutes
**SERVINGS:** 4 people
**DIFFICULTY:** easy
**INGREDIENTS:**
- 2 cups main ingredient
- 1 tbsp olive oil
- Salt and pepper to taste
**INSTRUCTIONS:**
1. Prepare all ingredients
2. Heat oil in pan over medium heat
3. Cook main ingredients for 20 minutes
4. Season with salt and pepper
5. Serve hot and enjoy!"""
    
    def _parse_recipe(self, generated_text: str, request: RecipeGenerationRequest) -> RecipeCreate:
        """Parse generated text into structured recipe format with improved parsing."""
        try:
            # Initialize with defaults
            title = "Generated Recipe"
            description = ""
            ingredients = []
            instructions = []
            prep_time = 15
            cook_time = 30
            servings = request.servings or 4
            difficulty = request.difficulty or "medium"
            
            # Parse structured format from modern LLM output
            lines = generated_text.split('\n')
            current_section = None
            ingredient_pattern = re.compile(r'^[-*•]\s*(.+)$')
            instruction_pattern = re.compile(r'^(\d+)\.\s*(.+)$')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Extract structured fields
                if line.startswith("**TITLE:**"):
                    title = line.replace("**TITLE:**", "").strip()
                elif line.startswith("**DESCRIPTION:**"):
                    description = line.replace("**DESCRIPTION:**", "").strip()
                elif line.startswith("**PREP TIME:**"):
                    prep_str = line.replace("**PREP TIME:**", "").strip()
                    prep_time = self._extract_time_minutes(prep_str)
                elif line.startswith("**COOK TIME:**"):
                    cook_str = line.replace("**COOK TIME:**", "").strip()
                    cook_time = self._extract_time_minutes(cook_str)
                elif line.startswith("**SERVINGS:**"):
                    serv_str = line.replace("**SERVINGS:**", "").strip()
                    servings = self._extract_servings(serv_str)
                elif line.startswith("**DIFFICULTY:**"):
                    difficulty = line.replace("**DIFFICULTY:**", "").strip().lower()
                elif line.startswith("**INGREDIENTS:**"):
                    current_section = "ingredients"
                    continue
                elif line.startswith("**INSTRUCTIONS:**"):
                    current_section = "instructions"
                    continue
                
                # Parse ingredients
                if current_section == "ingredients":
                    ingredient_match = ingredient_pattern.match(line)
                    if ingredient_match:
                        ingredient_text = ingredient_match.group(1)
                        parsed_ingredient = self._parse_ingredient_line(ingredient_text)
                        if parsed_ingredient:
                            ingredients.append(parsed_ingredient)
                
                # Parse instructions
                elif current_section == "instructions":
                    instruction_match = instruction_pattern.match(line)
                    if instruction_match:
                        instruction_text = instruction_match.group(2)
                        instructions.append(instruction_text)
                    elif line and not line.startswith("**"):
                        # Handle unnumbered instructions
                        instructions.append(line)
            
            # Fallback parsing if structured format failed
            if not ingredients or not instructions:
                return self._fallback_parse(generated_text, request)
            
            return RecipeCreate(
                title=title,
                description=description or f"A delicious {request.cuisine or ''} recipe".strip(),
                ingredients=ingredients,
                instructions=instructions,
                prep_time=prep_time,
                cook_time=cook_time,
                servings=servings,
                difficulty=difficulty,
                cuisine=request.cuisine,
                dietary_tags=request.dietary_restrictions or []
            )
            
        except Exception as e:
            logger.error(f"Recipe parsing failed: {e}")
            return self._create_fallback_recipe(request)
    
    def _extract_time_minutes(self, time_str: str) -> int:
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
    
    def _extract_servings(self, serv_str: str) -> int:
        """Extract serving count from string."""
        numbers = re.findall(r'\d+', serv_str)
        return int(numbers[0]) if numbers else 4
    
    def _parse_ingredient_line(self, ingredient_text: str) -> Optional[Dict]:
        """Parse ingredient line into structured format."""
        try:
            # Pattern to match "amount unit ingredient" format
            pattern = re.compile(r'^(\d+(?:\.\d+)?(?:/\d+)?)\s*(\w+)?\s+(.+)$')
            match = pattern.match(ingredient_text.strip())
            
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
                    "name": ingredient_text.strip(),
                    "amount": 1,
                    "unit": "piece",
                    "notes": None
                }
        except Exception:
            return {
                "name": ingredient_text.strip(),
                "amount": 1,
                "unit": "piece", 
                "notes": None
            }
    
    def _fallback_parse(self, generated_text: str, request: RecipeGenerationRequest) -> RecipeCreate:
        """Fallback parsing for unstructured text."""
        # Simple fallback based on request
        ingredients = []
        if request.ingredients:
            for ingredient in request.ingredients[:10]:
                ingredients.append({
                    "name": ingredient,
                    "amount": 1,
                    "unit": "cup",
                    "notes": None
                })
        
        instructions = [
            "Prepare all ingredients as needed.",
            "Follow cooking methods appropriate for the ingredients.",
            "Season to taste and serve."
        ]
        
        return RecipeCreate(
            title=f"{request.cuisine or 'Delicious'} Recipe",
            description="A tasty recipe generated for you.",
            ingredients=ingredients,
            instructions=instructions,
            prep_time=15,
            cook_time=30,
            servings=request.servings or 4,
            difficulty=request.difficulty or "medium",
            cuisine=request.cuisine,
            dietary_tags=request.dietary_restrictions or []
        )
    
    def _create_fallback_recipe(self, request: RecipeGenerationRequest) -> RecipeCreate:
        """Create a simple fallback recipe if generation fails."""
        ingredients_str = ", ".join(request.ingredients) if request.ingredients else "mixed vegetables"
        
        return RecipeCreate(
            title=f"Simple {request.cuisine or 'Home'} Style {ingredients_str.title()}",
            description="A simple and delicious recipe created for you.",
            ingredients=[
                {"name": ingredient, "amount": 1, "unit": "cup", "notes": None}
                for ingredient in (request.ingredients or ["vegetables", "olive oil", "salt"])
            ],
            instructions=[
                "Prepare all ingredients.",
                "Cook as desired.",
                "Season to taste.",
                "Serve and enjoy!"
            ],
            prep_time=10,
            cook_time=20,
            servings=request.servings or 4,
            difficulty=request.difficulty or "easy",
            cuisine=request.cuisine,
            dietary_tags=request.dietary_restrictions or []
        )
    
    async def generate_recipe_with_tools(self, request: RecipeGenerationRequest) -> Dict[str, Any]:
        """Generate a recipe with tool-enhanced capabilities."""
        try:
            # Step 1: Generate base recipe
            base_recipe = await self.generate_recipe(request)
            
            # Step 2: Use tools to enhance the recipe
            enhancements = await self._enhance_recipe_with_tools(base_recipe, request)
            
            return {
                "recipe": base_recipe,
                "enhancements": enhancements,
                "tool_enhanced": True
            }
            
        except Exception as e:
            logger.error(f"Tool-enhanced recipe generation failed: {e}")
            return {
                "recipe": await self.generate_recipe(request),
                "enhancements": {},
                "tool_enhanced": False,
                "error": str(e)
            }
    
    async def _enhance_recipe_with_tools(self, recipe: RecipeCreate, request: RecipeGenerationRequest) -> Dict[str, Any]:
        """Enhance recipe using available tools."""
        enhancements = {}
        
        try:
            # Calculate nutrition information
            if recipe.ingredients:
                nutrition_call = ToolCall(
                    name="nutrition_calculator",
                    arguments={
                        "ingredients": recipe.ingredients,
                        "servings": recipe.servings
                    },
                    call_id="nutrition_calc"
                )
                
                nutrition_result = await self.tool_system.execute_tool_call(nutrition_call)
                if nutrition_result.success:
                    enhancements["nutrition"] = nutrition_result.result
            
            # Calculate cooking times for ingredients
            cooking_times = []
            for ingredient in recipe.ingredients[:3]:  # Check first 3 ingredients
                time_call = ToolCall(
                    name="cooking_timer",
                    arguments={
                        "action": "calculate_time",
                        "cooking_method": "baking",  # Default method
                        "ingredient": ingredient["name"]
                    },
                    call_id=f"time_{ingredient['name']}"
                )
                
                time_result = await self.tool_system.execute_tool_call(time_call)
                if time_result.success and time_result.result.get("success"):
                    cooking_times.append({
                        "ingredient": ingredient["name"],
                        "estimated_time": time_result.result.get("estimated_minutes", 0)
                    })
            
            if cooking_times:
                enhancements["cooking_times"] = cooking_times
            
            # Suggest substitutions for dietary restrictions
            if request.dietary_restrictions and recipe.ingredients:
                substitutions = []
                for ingredient in recipe.ingredients[:3]:  # Check first 3 ingredients
                    sub_call = ToolCall(
                        name="ingredient_substitution",
                        arguments={
                            "ingredient": ingredient["name"],
                            "dietary_restrictions": request.dietary_restrictions,
                            "recipe_context": "general cooking"
                        },
                        call_id=f"sub_{ingredient['name']}"
                    )
                    
                    sub_result = await self.tool_system.execute_tool_call(sub_call)
                    if sub_result.success and sub_result.result.get("success"):
                        substitutions.append({
                            "original": ingredient["name"],
                            "substitutions": sub_result.result.get("substitutions", {})
                        })
                
                if substitutions:
                    enhancements["substitutions"] = substitutions
            
            # Scale recipe if different serving size requested
            if request.servings and request.servings != recipe.servings:
                scale_call = ToolCall(
                    name="recipe_scaler",
                    arguments={
                        "ingredients": recipe.ingredients,
                        "original_servings": recipe.servings,
                        "target_servings": request.servings
                    },
                    call_id="recipe_scale"
                )
                
                scale_result = await self.tool_system.execute_tool_call(scale_call)
                if scale_result.success:
                    enhancements["scaled_ingredients"] = scale_result.result
            
            return enhancements
            
        except Exception as e:
            logger.error(f"Recipe enhancement with tools failed: {e}")
            return {"error": str(e)}