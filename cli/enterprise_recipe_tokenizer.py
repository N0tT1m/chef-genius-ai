#!/usr/bin/env python3
"""
🏢 ENTERPRISE-GRADE RECIPE TOKENIZER
State-of-the-art tokenizer designed for commercial recipe AI systems

Features:
- 500+ culinary-specific special tokens
- International cuisine support
- Nutritional and dietary awareness
- Professional cooking terminology
- Equipment and technique vocabulary
- Temperature, time, and measurement precision
"""

from transformers import AutoTokenizer
import json
from typing import Dict, List, Tuple, Optional
from pathlib import Path

class EnterpriseRecipeTokenizer:
    """Enterprise-grade tokenizer for recipe generation AI systems."""
    
    def __init__(self, base_model: str = "google/flan-t5-large"):
        self.base_model = base_model
        self.tokenizer = None
        self.recipe_vocabulary = self._build_culinary_vocabulary()
        self.special_tokens = self._create_special_tokens()
        
    def _build_culinary_vocabulary(self) -> Dict[str, List[str]]:
        """Build comprehensive culinary vocabulary for enterprise and B2B use."""
        
        vocabulary = {
            # === COOKING METHODS & TECHNIQUES ===
            "cooking_methods": [
                "ROAST", "BAKE", "GRILL", "SAUTE", "BRAISE", "STEW", "SIMMER", "BOIL",
                "STEAM", "POACH", "FRY", "DEEP_FRY", "STIR_FRY", "PAN_FRY", "SEAR",
                "CHAR", "SMOKE", "CURE", "PICKLE", "MARINATE", "BLANCH", "PARBOIL",
                "SOUS_VIDE", "CONFIT", "FLAMBE", "CARAMELIZE", "REDUCE", "DEGLAZE",
                "EMULSIFY", "WHIP", "FOLD", "KNEAD", "PROOF", "FERMENT", "TEMPER",
                # Commercial techniques
                "BATCH_COOK", "PREP_AHEAD", "HOT_HOLD", "COLD_HOLD", "BLAST_CHILL",
                "VACUUM_PACK", "PORTION_CONTROL", "STANDARDIZE", "SCALE_UP", "BULK_PREP"
            ],
            
            # === PREPARATION TECHNIQUES ===
            "prep_techniques": [
                "CHOP", "DICE", "MINCE", "SLICE", "JULIENNE", "CHIFFONADE", "BRUNOISE",
                "ROUGH_CHOP", "FINE_DICE", "MEDIUM_DICE", "LARGE_DICE", "BIAS_CUT",
                "MANDOLINE", "SHRED", "GRATE", "ZEST", "PEEL", "CORE", "SEED",
                "TRIM", "FILLET", "DEBONE", "BUTTERFLY", "POUND", "TENDERIZE"
            ],
            
            # === MEASUREMENTS & QUANTITIES ===
            "measurements": [
                "TSP", "TBSP", "CUP", "PINT", "QUART", "GALLON", "FLUID_OZ", "ML", "L",
                "G", "KG", "OZ", "LB", "PINCH", "DASH", "SPLASH", "HANDFUL",
                "CLOVE", "HEAD", "BUNCH", "SPRIG", "STALK", "PIECE", "SLICE",
                "CAN", "JAR", "BOTTLE", "PACKAGE", "BAG", "BOX"
            ],
            
            # === TEMPERATURES ===
            "temperatures": [
                "ROOM_TEMP", "COLD", "CHILLED", "FROZEN", "COOL", "LUKEWARM", "WARM",
                "HOT", "BOILING", "SIMMERING", "SCALDING", "LOW_HEAT", "MEDIUM_LOW",
                "MEDIUM_HEAT", "MEDIUM_HIGH", "HIGH_HEAT", "VERY_HOT",
                "DEGREES_F", "DEGREES_C", "CELSIUS", "FAHRENHEIT"
            ],
            
            # === TIMING ===
            "timing": [
                "SECOND", "MINUTE", "HOUR", "OVERNIGHT", "QUICK", "SLOW", "RAPID",
                "INSTANT", "IMMEDIATE", "GRADUAL", "UNTIL_DONE", "AL_DENTE",
                "TENDER", "CRISP", "GOLDEN", "CARAMELIZED", "REDUCED"
            ],
            
            # === EQUIPMENT ===
            "equipment": [
                "OVEN", "STOVETOP", "MICROWAVE", "GRILL", "SMOKER", "FRYER",
                "SLOW_COOKER", "PRESSURE_COOKER", "SOUS_VIDE", "STEAMER",
                "FOOD_PROCESSOR", "BLENDER", "MIXER", "IMMERSION_BLENDER",
                "MANDOLINE", "GRATER", "ZESTER", "THERMOMETER", "SCALE",
                "KNIFE", "CUTTING_BOARD", "PAN", "POT", "SKILLET", "WOK",
                "DUTCH_OVEN", "ROASTING_PAN", "BAKING_SHEET", "CAKE_PAN"
            ],
            
            # === INTERNATIONAL CUISINES ===
            "cuisines": [
                "ITALIAN", "FRENCH", "CHINESE", "JAPANESE", "KOREAN", "THAI",
                "VIETNAMESE", "INDIAN", "MEXICAN", "SPANISH", "GREEK", "TURKISH",
                "MOROCCAN", "ETHIOPIAN", "PERUVIAN", "BRAZILIAN", "ARGENTINIAN",
                "AMERICAN", "SOUTHERN", "CAJUN", "CREOLE", "BBQ", "SOUL_FOOD"
            ],
            
            # === DIETARY CATEGORIES ===
            "dietary": [
                "VEGAN", "VEGETARIAN", "PESCATARIAN", "KETO", "PALEO", "LOW_CARB",
                "GLUTEN_FREE", "DAIRY_FREE", "NUT_FREE", "SOY_FREE", "KOSHER",
                "HALAL", "WHOLE30", "MEDITERRANEAN", "DIABETIC_FRIENDLY",
                "HEART_HEALTHY", "LOW_SODIUM", "HIGH_PROTEIN", "LOW_FAT"
            ],
            
            # === FOOD CATEGORIES ===
            "food_categories": [
                "PROTEIN", "MEAT", "POULTRY", "SEAFOOD", "FISH", "SHELLFISH",
                "DAIRY", "CHEESE", "EGGS", "VEGETABLE", "FRUIT", "GRAIN",
                "LEGUME", "NUT", "SEED", "HERB", "SPICE", "CONDIMENT",
                "SAUCE", "OIL", "VINEGAR", "STOCK", "BROTH"
            ],
            
            # === TEXTURES & DESCRIPTIONS ===
            "textures": [
                "CRISPY", "CRUNCHY", "TENDER", "JUICY", "MOIST", "DRY", "FLAKY",
                "SMOOTH", "CREAMY", "CHUNKY", "THICK", "THIN", "LIGHT", "HEAVY",
                "AIRY", "DENSE", "CHEWY", "SOFT", "FIRM", "SILKY", "VELVETY"
            ],
            
            # === FLAVORS ===
            "flavors": [
                "SWEET", "SALTY", "SOUR", "BITTER", "UMAMI", "SPICY", "HOT",
                "MILD", "TANGY", "ZESTY", "RICH", "BRIGHT", "EARTHY", "SMOKY",
                "FRESH", "AROMATIC", "FRAGRANT", "PUNGENT", "DELICATE", "BOLD"
            ],
            
            # === NUTRITIONAL TERMS ===
            "nutrition": [
                "CALORIES", "PROTEIN", "CARBS", "FAT", "FIBER", "SUGAR", "SODIUM",
                "VITAMIN", "MINERAL", "ANTIOXIDANT", "OMEGA_3", "PROBIOTIC",
                "HEALTHY", "NUTRITIOUS", "BALANCED", "PORTION", "SERVING"
            ],
            
            # === B2B BUSINESS TERMS ===
            "business_operations": [
                "FOOD_COST", "LABOR_COST", "PORTION_SIZE", "YIELD", "WASTE_REDUCTION",
                "INVENTORY", "SHELF_LIFE", "SCALABLE", "STANDARDIZED", "CONSISTENT",
                "PROFITABLE", "EFFICIENT", "WORKFLOW", "PRODUCTIVITY", "TURNOVER",
                "COVERS", "VOLUME", "BATCH_SIZE", "PREP_TIME", "SERVICE_TIME"
            ],
            
            # === COMMERCIAL EQUIPMENT ===
            "commercial_equipment": [
                "CONVECTION_OVEN", "COMBI_OVEN", "STEAM_TABLE", "BLAST_CHILLER",
                "CHAR_BROILER", "FLAT_TOP_GRILL", "SALAMANDER", "FRYER_STATION",
                "STEAM_KETTLE", "TILT_SKILLET", "MIXER_PADDLE", "FOOD_PROCESSOR_COMMERCIAL",
                "SLICER_COMMERCIAL", "VACUUM_CHAMBER", "IMMERSION_CIRCULATOR",
                "HOLDING_CABINET", "SPEED_OVEN", "PRESSURE_STEAMER", "INDUCTION_COOKTOP"
            ],
            
            # === SERVICE STYLES ===
            "service_styles": [
                "FINE_DINING", "CASUAL_DINING", "FAST_CASUAL", "QUICK_SERVICE",
                "BUFFET_STYLE", "FAMILY_STYLE", "PLATED_SERVICE", "BANQUET_STYLE",
                "CAFETERIA_STYLE", "GRAB_AND_GO", "DELIVERY_OPTIMIZED", "TAKEOUT_FRIENDLY",
                "CATERING_STYLE", "INSTITUTIONAL", "CORPORATE_DINING"
            ],
            
            # === VENUE TYPES ===
            "venue_types": [
                "RESTAURANT", "CATERING", "HOTEL", "HOSPITAL", "SCHOOL", "CORPORATE",
                "SENIOR_LIVING", "PRISON", "MILITARY", "AIRLINE", "STADIUM", "ARENA",
                "CASINO", "RESORT", "CRUISE_SHIP", "FOOD_TRUCK", "GHOST_KITCHEN",
                "MEAL_KIT", "COMMISSARY", "CENTRAL_KITCHEN"
            ],
            
            # === COST MANAGEMENT ===
            "cost_terms": [
                "BUDGET", "PREMIUM", "LUXURY", "ECONOMICAL", "VALUE", "COST_EFFECTIVE",
                "MARGIN", "MARKUP", "FOOD_COST_PERCENTAGE", "LABOR_PERCENTAGE",
                "PRIME_COST", "BREAK_EVEN", "PROFIT_MARGIN", "UNIT_COST", "TOTAL_COST"
            ],
            
            # === FOOD SAFETY & COMPLIANCE ===
            "food_safety": [
                "HACCP", "TEMPERATURE_CONTROL", "CRITICAL_CONTROL_POINT", "ALLERGEN_FREE",
                "CROSS_CONTAMINATION", "SANITIZE", "PERSONAL_HYGIENE", "STORAGE_TEMP",
                "HOLDING_TEMP", "REHEATING_TEMP", "COOLING_PROCEDURE", "THAWING_SAFE",
                "SHELF_STABLE", "REFRIGERATED", "FROZEN", "AMBIENT_STABLE"
            ],
            
            # === SCALING & VOLUME ===
            "scaling_terms": [
                "SINGLE_SERVING", "FAMILY_SIZE", "BATCH_COOKING", "LARGE_BATCH",
                "INDUSTRIAL_SCALE", "INDIVIDUAL_PORTION", "BULK_PREPARATION",
                "UNIT_SCALING", "RECIPE_CONVERSION", "YIELD_CALCULATION",
                "PRODUCTION_VOLUME", "CAPACITY_PLANNING", "THROUGHPUT"
            ],
            
            # === DIETARY EXPANSION ===
            "advanced_dietary": [
                "ANTI_INFLAMMATORY", "AUTOIMMUNE_PROTOCOL", "CARNIVORE", "FODMAP_LOW",
                "LECTIN_FREE", "NIGHTSHADE_FREE", "HISTAMINE_LOW", "OXALATE_LOW",
                "DIABETIC_FRIENDLY", "RENAL_DIET", "CARDIAC_DIET", "PUREED_TEXTURE",
                "MINCED_TEXTURE", "SOFT_MECHANICAL", "CLEAR_LIQUID", "FULL_LIQUID"
            ],
            
            # === INTERNATIONAL EXPANSION ===
            "regional_cuisines": [
                "NORDIC", "SCANDINAVIAN", "EASTERN_EUROPEAN", "MIDDLE_EASTERN",
                "NORTH_AFRICAN", "WEST_AFRICAN", "CARIBBEAN", "CENTRAL_AMERICAN",
                "SOUTH_AMERICAN", "SOUTHEAST_ASIAN", "CENTRAL_ASIAN", "POLYNESIAN",
                "FUSION", "MODERN_AMERICAN", "NEW_WORLD", "OLD_WORLD"
            ],
            
            # === TEXTURE & PRESENTATION ===
            "advanced_textures": [
                "MOLECULAR", "SPHERIFICATION", "GELIFICATION", "FOAMING", "CARBONATED",
                "DEHYDRATED", "FREEZE_DRIED", "CRYSTALLIZED", "ENCAPSULATED",
                "LAYERED", "COMPOSED", "RUSTIC", "REFINED", "ARTISANAL", "INDUSTRIAL"
            ],
            
            # === SUSTAINABILITY ===
            "sustainability": [
                "LOCAL_SOURCED", "SEASONAL", "ORGANIC", "SUSTAINABLE", "FAIR_TRADE",
                "CARBON_NEUTRAL", "ZERO_WASTE", "UPCYCLED", "PLANT_FORWARD",
                "REGENERATIVE", "FARM_TO_TABLE", "NOSE_TO_TAIL", "ROOT_TO_STEM",
                "ENVIRONMENTALLY_FRIENDLY", "ETHICALLY_SOURCED"
            ],
            
            # === MEAL STRUCTURE ===
            "meal_structure": [
                "PROTEIN_CENTRIC", "ONE_PROTEIN_TWO_SIDES", "TRIO_MEAL", "COMPOSED_PLATE",
                "FAMILY_MEAL", "INDIVIDUAL_MEAL", "SHAREABLE", "TAPAS_STYLE",
                "MULTI_COURSE", "SINGLE_COURSE", "MODULAR_MEAL", "BUILD_YOUR_OWN",
                "PRESET_COMBINATION", "CUSTOMIZABLE", "STANDARDIZED_PORTION"
            ]
        }
        
        return vocabulary
    
    def _create_special_tokens(self) -> List[str]:
        """Create comprehensive special tokens for recipe structure."""
        
        # === STRUCTURAL TOKENS ===
        structural = [
            "[RECIPE_START]", "[RECIPE_END]",
            "[TITLE_START]", "[TITLE_END]", 
            "[DESCRIPTION_START]", "[DESCRIPTION_END]",
            "[SERVINGS_START]", "[SERVINGS_END]",
            "[PREP_TIME_START]", "[PREP_TIME_END]",
            "[COOK_TIME_START]", "[COOK_TIME_END]",
            "[TOTAL_TIME_START]", "[TOTAL_TIME_END]",
            "[DIFFICULTY_START]", "[DIFFICULTY_END]",
            "[CUISINE_START]", "[CUISINE_END]",
            "[DIETARY_START]", "[DIETARY_END]",
            "[INGREDIENTS_START]", "[INGREDIENTS_END]",
            "[INSTRUCTIONS_START]", "[INSTRUCTIONS_END]",
            "[NUTRITION_START]", "[NUTRITION_END]",
            "[TIPS_START]", "[TIPS_END]",
            "[VARIATIONS_START]", "[VARIATIONS_END]",
            "[EQUIPMENT_START]", "[EQUIPMENT_END]",
            "[NOTES_START]", "[NOTES_END]"
        ]
        
        # === B2B & COMMERCIAL TOKENS ===
        b2b_tokens = [
            "[BUSINESS_TYPE]", "[VENUE_TYPE]", "[SERVICE_STYLE]",
            "[VOLUME_TARGET]", "[COST_TARGET]", "[SKILL_LEVEL]",
            "[FOOD_COST]", "[LABOR_COST]", "[PROFIT_MARGIN]",
            "[BATCH_SIZE]", "[SCALING_NOTES]", "[WORKFLOW]",
            "[FOOD_SAFETY]", "[ALLERGEN_INFO]", "[STORAGE_NOTES]",
            "[EQUIPMENT_COMMERCIAL]", "[PREP_AHEAD]", "[HOLDING_INSTRUCTIONS]",
            "[PORTION_CONTROL]", "[YIELD_INFO]", "[WASTE_REDUCTION]",
            "[STAFF_TRAINING]", "[STANDARDIZATION]", "[QUALITY_CONTROL]"
        ]
        
        # === MEAL STRUCTURE TOKENS ===
        meal_structure_tokens = [
            "[PRIMARY_PROTEIN]", "[SIDE_ONE]", "[SIDE_TWO]", "[SIDE_THREE]",
            "[PROTEIN_PORTION]", "[SIDE_PORTION]", "[GARNISH]", "[SAUCE]",
            "[MEAL_COMPOSITION]", "[PLATE_LAYOUT]", "[PRESENTATION_NOTES]",
            "[TRIO_MEAL]", "[FAMILY_STYLE]", "[INDIVIDUAL_PORTION]"
        ]
        
        # === EDGE CASE TOKENS ===
        edge_case_tokens = [
            "[DIETARY_RESTRICTION]", "[ALLERGEN_WARNING]", "[SUBSTITUTION]",
            "[EMERGENCY_SUBSTITUTION]", "[COST_CONSTRAINT]", "[TIME_CONSTRAINT]",
            "[EQUIPMENT_LIMITATION]", "[SKILL_LIMITATION]", "[VOLUME_CONSTRAINT]",
            "[SEASONAL_LIMITATION]", "[SUPPLY_CHAIN_ISSUE]", "[BUDGET_OVERRIDE]",
            "[URGENT_REQUEST]", "[CUSTOM_MODIFICATION]", "[SPECIAL_OCCASION]"
        ]
        
        # === EXPANSION TOKENS ===
        expansion_tokens = [
            "[FRANCHISE_READY]", "[MULTI_LOCATION]", "[CENTRAL_KITCHEN]",
            "[GHOST_KITCHEN]", "[DELIVERY_OPTIMIZED]", "[TAKEOUT_FRIENDLY]",
            "[MEAL_KIT_READY]", "[RETAIL_READY]", "[FROZEN_READY]",
            "[SHELF_STABLE]", "[INTERNATIONAL_EXPANSION]", "[REGIONAL_ADAPTATION]",
            "[CULTURAL_SENSITIVITY]", "[LOCAL_INGREDIENTS]", "[SEASONAL_MENU]"
        ]
        
        # === INGREDIENT TOKENS ===
        ingredient_tokens = [
            "[INGREDIENT]", "[QUANTITY]", "[UNIT]", "[PREPARATION]",
            "[OPTIONAL]", "[SUBSTITUTE]", "[BRAND]", "[QUALITY]"
        ]
        
        # === INSTRUCTION TOKENS ===
        instruction_tokens = [
            "[STEP]", "[SUBSTEP]", "[WARNING]", "[TIP]", "[TIMING]",
            "[TEMPERATURE]", "[VISUAL_CUE]", "[TECHNIQUE]", "[EQUIPMENT_NEEDED]"
        ]
        
        # === CULINARY TECHNIQUE TOKENS ===
        technique_tokens = []
        for category, terms in self.recipe_vocabulary.items():
            for term in terms:
                technique_tokens.append(f"[{term}]")
        
        # Combine all tokens
        all_tokens = (structural + b2b_tokens + meal_structure_tokens + 
                     edge_case_tokens + expansion_tokens + ingredient_tokens + 
                     instruction_tokens + technique_tokens)
        
        return all_tokens
    
    def create_tokenizer(self) -> Tuple[AutoTokenizer, int]:
        """Create the enhanced tokenizer with all culinary tokens."""
        
        print("🏢 Creating Enterprise Recipe Tokenizer...")
        print(f"📚 Base model: {self.base_model}")
        
        # Load base tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        
        # Add special tokens
        special_tokens_dict = {
            "additional_special_tokens": self.special_tokens
        }
        
        num_added = self.tokenizer.add_special_tokens(special_tokens_dict)
        
        # Ensure proper padding
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"✅ Added {num_added} culinary-specific tokens")
        print(f"📊 Total vocabulary size: {len(self.tokenizer):,}")
        print(f"🎯 Culinary categories: {len(self.recipe_vocabulary)}")
        
        return self.tokenizer, num_added
    
    def smart_truncate_prompt(self, prompt: str, max_tokens: int = 450) -> str:
        """Intelligently truncate prompts to fit model limits while preserving key information."""
        
        if self.tokenizer is None:
            raise ValueError("Tokenizer not created yet. Call create_tokenizer() first.")
        
        # First check if truncation is needed
        tokens = self.tokenizer.encode(prompt)
        if len(tokens) <= max_tokens:
            return prompt
        
        print(f"⚠️  Prompt too long ({len(tokens)} tokens), truncating to {max_tokens}...")
        
        # Split prompt into sections
        sections = prompt.split('\n\n')
        
        # Priority order for sections (keep most important)
        priority_keywords = [
            'User Request:', 'Commercial Recipe:', 'Recipe:',  # Always keep
            '[RECIPE_START]', '[TITLE_START]', '[INGREDIENTS_START]', '[INSTRUCTIONS_START]',  # Structure
            '[BUSINESS_TYPE]', '[COST_TARGET]', '[VOLUME_TARGET]',  # B2B essentials
            'COMMERCIAL RECIPE GENERATION REQUEST',  # Header
        ]
        
        # Separate high-priority and low-priority sections
        essential_sections = []
        optional_sections = []
        
        for section in sections:
            is_essential = any(keyword in section for keyword in priority_keywords)
            if is_essential:
                essential_sections.append(section)
            else:
                optional_sections.append(section)
        
        # Start with essential sections
        truncated_prompt = '\n\n'.join(essential_sections)
        
        # Add optional sections if space allows
        for section in optional_sections:
            test_prompt = truncated_prompt + '\n\n' + section
            test_tokens = self.tokenizer.encode(test_prompt)
            
            if len(test_tokens) <= max_tokens:
                truncated_prompt = test_prompt
            else:
                break
        
        # Final check and hard truncation if needed
        final_tokens = self.tokenizer.encode(truncated_prompt)
        if len(final_tokens) > max_tokens:
            # Hard truncate by decoding subset of tokens
            truncated_tokens = final_tokens[:max_tokens-10]  # Leave space for generation
            truncated_prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
        
        final_token_count = len(self.tokenizer.encode(truncated_prompt))
        print(f"✅ Truncated to {final_token_count} tokens")
        
        return truncated_prompt
    
    def format_enterprise_recipe(self, recipe_data: Dict) -> str:
        """Format recipe data using enterprise tokenizer structure."""
        
        formatted = "[RECIPE_START]\n"
        
        # Title
        if 'title' in recipe_data:
            formatted += f"[TITLE_START]{recipe_data['title']}[TITLE_END]\n"
        
        # Description
        if 'description' in recipe_data:
            formatted += f"[DESCRIPTION_START]{recipe_data['description']}[DESCRIPTION_END]\n"
        
        # Metadata
        if 'servings' in recipe_data:
            formatted += f"[SERVINGS_START]{recipe_data['servings']}[SERVINGS_END]\n"
        
        if 'prep_time' in recipe_data:
            formatted += f"[PREP_TIME_START]{recipe_data['prep_time']}[PREP_TIME_END]\n"
            
        if 'cook_time' in recipe_data:
            formatted += f"[COOK_TIME_START]{recipe_data['cook_time']}[COOK_TIME_END]\n"
            
        if 'difficulty' in recipe_data:
            formatted += f"[DIFFICULTY_START]{recipe_data['difficulty']}[DIFFICULTY_END]\n"
            
        if 'cuisine' in recipe_data:
            formatted += f"[CUISINE_START]{recipe_data['cuisine']}[CUISINE_END]\n"
            
        if 'dietary' in recipe_data:
            formatted += f"[DIETARY_START]{recipe_data['dietary']}[DIETARY_END]\n"
        
        # Equipment
        if 'equipment' in recipe_data:
            formatted += f"\n[EQUIPMENT_START]\n"
            if isinstance(recipe_data['equipment'], list):
                for item in recipe_data['equipment']:
                    formatted += f"• {item}\n"
            else:
                formatted += f"{recipe_data['equipment']}\n"
            formatted += f"[EQUIPMENT_END]\n"
        
        # Ingredients with detailed structure
        formatted += f"\n[INGREDIENTS_START]\n"
        if 'ingredients' in recipe_data:
            if isinstance(recipe_data['ingredients'], list):
                for ingredient in recipe_data['ingredients']:
                    if isinstance(ingredient, dict):
                        # Structured ingredient
                        line = "[INGREDIENT]"
                        if 'quantity' in ingredient:
                            line += f"[QUANTITY]{ingredient['quantity']}[/QUANTITY]"
                        if 'unit' in ingredient:
                            line += f"[UNIT]{ingredient['unit']}[/UNIT]"
                        if 'name' in ingredient:
                            line += f"{ingredient['name']}"
                        if 'preparation' in ingredient:
                            line += f"[PREPARATION]{ingredient['preparation']}[/PREPARATION]"
                        line += "[/INGREDIENT]"
                        formatted += f"{line}\n"
                    else:
                        # Simple ingredient string
                        formatted += f"[INGREDIENT]{ingredient}[/INGREDIENT]\n"
            else:
                formatted += f"{recipe_data['ingredients']}\n"
        formatted += f"[INGREDIENTS_END]\n"
        
        # Instructions with step structure
        formatted += f"\n[INSTRUCTIONS_START]\n"
        if 'instructions' in recipe_data:
            if isinstance(recipe_data['instructions'], list):
                for i, instruction in enumerate(recipe_data['instructions'], 1):
                    if isinstance(instruction, dict):
                        # Structured instruction
                        formatted += f"[STEP]{i}[/STEP]"
                        if 'technique' in instruction:
                            formatted += f"[TECHNIQUE]{instruction['technique']}[/TECHNIQUE]"
                        if 'timing' in instruction:
                            formatted += f"[TIMING]{instruction['timing']}[/TIMING]"
                        if 'temperature' in instruction:
                            formatted += f"[TEMPERATURE]{instruction['temperature']}[/TEMPERATURE]"
                        formatted += f"{instruction.get('text', instruction.get('step', ''))}\n"
                    else:
                        # Simple instruction string
                        formatted += f"[STEP]{i}[/STEP]{instruction}\n"
            else:
                formatted += f"{recipe_data['instructions']}\n"
        formatted += f"[INSTRUCTIONS_END]\n"
        
        # Nutrition
        if 'nutrition' in recipe_data:
            formatted += f"\n[NUTRITION_START]\n{recipe_data['nutrition']}\n[NUTRITION_END]\n"
        
        # Tips
        if 'tips' in recipe_data:
            formatted += f"\n[TIPS_START]\n{recipe_data['tips']}\n[TIPS_END]\n"
        
        # Variations
        if 'variations' in recipe_data:
            formatted += f"\n[VARIATIONS_START]\n{recipe_data['variations']}\n[VARIATIONS_END]\n"
        
        formatted += "[RECIPE_END]"
        
        return formatted
    
    def create_enterprise_prompt(self, user_request: str, **kwargs) -> str:
        """Create enterprise-grade prompts for recipe generation."""
        
        # Extract parameters
        servings = kwargs.get('servings', '4')
        dietary = kwargs.get('dietary', None)
        cuisine = kwargs.get('cuisine', None)
        difficulty = kwargs.get('difficulty', None)
        max_time = kwargs.get('max_time', None)
        equipment = kwargs.get('equipment', None)
        
        prompt = f"""Generate a complete, detailed recipe following this exact structure:

[RECIPE_START]
[TITLE_START]Recipe Title[TITLE_END]
[DESCRIPTION_START]Brief, appetizing description[DESCRIPTION_END]
[SERVINGS_START]{servings}[SERVINGS_END]
[PREP_TIME_START]X minutes[PREP_TIME_END]
[COOK_TIME_START]X minutes[COOK_TIME_END]
[DIFFICULTY_START]Easy/Medium/Hard[DIFFICULTY_END]"""

        if cuisine:
            prompt += f"\n[CUISINE_START]{cuisine}[CUISINE_END]"
        if dietary:
            prompt += f"\n[DIETARY_START]{dietary}[DIETARY_START]"
        
        prompt += f"""

[EQUIPMENT_START]
• List required equipment
[EQUIPMENT_END]

[INGREDIENTS_START]
[INGREDIENT][QUANTITY]amount[/QUANTITY][UNIT]unit[/UNIT]ingredient name[PREPARATION]preparation method[/PREPARATION][/INGREDIENT]
[INGREDIENT]ingredient 2[/INGREDIENT]
[INGREDIENT]ingredient 3[/INGREDIENT]
[INGREDIENTS_END]

[INSTRUCTIONS_START]
[STEP]1[/STEP][TECHNIQUE]cooking technique[/TECHNIQUE][TIMING]time estimate[/TIMING]Detailed step description
[STEP]2[/STEP]Next step with specific details
[STEP]3[/STEP]Continue with clear instructions
[INSTRUCTIONS_END]

[TIPS_START]
Professional cooking tips and suggestions
[TIPS_END]
[RECIPE_END]

User Request: {user_request}

Recipe:"""
        
        return prompt
    
    def create_b2b_prompt(self, user_request: str, **kwargs) -> str:
        """Create B2B-specific prompts for commercial recipe generation."""
        
        # B2B-specific parameters
        business_type = kwargs.get('business_type', 'Restaurant')
        service_style = kwargs.get('service_style', 'Casual Dining')
        volume = kwargs.get('volume', '100 servings')
        cost_target = kwargs.get('cost_target', 'Mid-range')
        skill_level = kwargs.get('skill_level', 'Line Cook')
        meal_structure = kwargs.get('meal_structure', '1 protein + 2 sides')
        
        # Standard parameters
        servings = kwargs.get('servings', '4')
        dietary = kwargs.get('dietary', None)
        cuisine = kwargs.get('cuisine', None)
        
        # Create compact B2B prompt
        prompt = f"""Generate commercial recipe: {user_request}

[BUSINESS_TYPE]{business_type}[/BUSINESS_TYPE] [SERVICE_STYLE]{service_style}[/SERVICE_STYLE] [VOLUME_TARGET]{volume}[/VOLUME_TARGET] [COST_TARGET]{cost_target}[/COST_TARGET] [SKILL_LEVEL]{skill_level}[/SKILL_LEVEL] [MEAL_COMPOSITION]{meal_structure}[/MEAL_COMPOSITION]

[RECIPE_START]
[TITLE_START]Recipe Title[TITLE_END]
[SERVINGS_START]{servings}[SERVINGS_START]
[PREP_TIME_START]X min[PREP_TIME_END]
[COOK_TIME_START]X min[COOK_TIME_END]"""

        if cuisine:
            prompt += f" [CUISINE_START]{cuisine}[CUISINE_END]"
        if dietary:
            prompt += f" [DIETARY_START]{dietary}[DIETARY_END]"

        prompt += f"""

[INGREDIENTS_START]
[PRIMARY_PROTEIN][QUANTITY]amount[/QUANTITY][UNIT]unit[/UNIT]protein[/PRIMARY_PROTEIN]
[SIDE_ONE]first side ingredients[/SIDE_ONE]
[SIDE_TWO]second side ingredients[/SIDE_TWO]
[INGREDIENTS_END]

[INSTRUCTIONS_START]
[STEP]1[/STEP][TECHNIQUE]method[/TECHNIQUE]Prepare protein with commercial techniques
[STEP]2[/STEP]Prepare first side optimized for volume
[STEP]3[/STEP]Prepare second side with timing coordination  
[STEP]4[/STEP][PORTION_CONTROL]portion control[/PORTION_CONTROL]Assembly and service
[INSTRUCTIONS_END]

[TIPS_START]Commercial tips and cost control[TIPS_END]
[RECIPE_END]

Recipe:"""
        
        # Apply smart truncation
        return self.smart_truncate_prompt(prompt, max_tokens=450)
    
    def create_edge_case_prompt(self, user_request: str, edge_case_type: str, **kwargs) -> str:
        """Create prompts for edge case testing scenarios."""
        
        constraint_prompts = {
            "dietary_restrictions": "[DIETARY_RESTRICTION]Multiple restrictions[/DIETARY_RESTRICTION]",
            "cost_constraints": "[COST_CONSTRAINT]Ultra budget[/COST_CONSTRAINT]", 
            "time_constraints": "[TIME_CONSTRAINT]Limited time[/TIME_CONSTRAINT]",
            "equipment_limitations": "[EQUIPMENT_LIMITATION]Basic equipment[/EQUIPMENT_LIMITATION]",
            "volume_constraints": "[VOLUME_CONSTRAINT]Unusual volume[/VOLUME_CONSTRAINT]"
        }
        
        # Create compact edge case prompt
        edge_constraint = constraint_prompts.get(edge_case_type, "")
        
        prompt = f"""Generate recipe with constraint: {user_request}
{edge_constraint}

[RECIPE_START]
[TITLE_START]Recipe Title[TITLE_END]
[INGREDIENTS_START]
[PRIMARY_PROTEIN]protein[/PRIMARY_PROTEIN]
[SIDE_ONE]side 1[/SIDE_ONE]
[SIDE_TWO]side 2[/SIDE_TWO]
[INGREDIENTS_END]

[INSTRUCTIONS_START]
[STEP]1[/STEP]Prepare protein considering constraint
[STEP]2[/STEP]Prepare sides with workaround
[STEP]3[/STEP]Assembly with constraint solution
[INSTRUCTIONS_END]
[RECIPE_END]

Recipe:"""
        
        return self.smart_truncate_prompt(prompt, max_tokens=400)
    
    def create_expansion_prompt(self, user_request: str, expansion_type: str, **kwargs) -> str:
        """Create prompts for business expansion scenarios."""
        
        expansion_prompts = {
            "franchise": "[FRANCHISE_READY]Standardizable[/FRANCHISE_READY]",
            "international": "[INTERNATIONAL_EXPANSION]Market adaptable[/INTERNATIONAL_EXPANSION]",
            "meal_kit": "[MEAL_KIT_READY]Home cooking[/MEAL_KIT_READY]",
            "ghost_kitchen": "[GHOST_KITCHEN]Delivery optimized[/GHOST_KITCHEN]"
        }
        
        expansion_constraint = expansion_prompts.get(expansion_type, "")
        
        prompt = f"""Generate expansion recipe: {user_request}
{expansion_constraint}

[RECIPE_START]
[TITLE_START]Scalable Recipe Title[TITLE_END]
[INGREDIENTS_START]
[PRIMARY_PROTEIN]protein[/PRIMARY_PROTEIN]
[SIDE_ONE]side 1[/SIDE_ONE]
[SIDE_TWO]side 2[/SIDE_TWO]
[INGREDIENTS_END]

[INSTRUCTIONS_START]
[STEP]1[/STEP]Prepare protein for expansion model
[STEP]2[/STEP]Prepare sides with scalability
[STEP]3[/STEP]Assembly for {expansion_type} requirements
[INSTRUCTIONS_END]

[SCALING_NOTES]Expansion considerations[/SCALING_NOTES]
[RECIPE_END]

Recipe:"""
        
        return self.smart_truncate_prompt(prompt, max_tokens=400)
    
    def save_tokenizer(self, save_path: str):
        """Save the enhanced tokenizer."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not created yet. Call create_tokenizer() first.")
        
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(str(save_dir))
        
        # Save metadata
        metadata = {
            "base_model": self.base_model,
            "vocabulary_categories": list(self.recipe_vocabulary.keys()),
            "total_special_tokens": len(self.special_tokens),
            "vocabulary_size": len(self.tokenizer)
        }
        
        with open(save_dir / "tokenizer_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Enterprise tokenizer saved to: {save_dir}")
        
    def get_statistics(self) -> Dict:
        """Get comprehensive tokenizer statistics."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not created yet.")
        
        stats = {
            "base_model": self.base_model,
            "total_vocabulary_size": len(self.tokenizer),
            "special_tokens_added": len(self.special_tokens),
            "culinary_categories": len(self.recipe_vocabulary),
            "category_breakdown": {
                category: len(terms) 
                for category, terms in self.recipe_vocabulary.items()
            }
        }
        
        return stats

def demo_enterprise_tokenizer():
    """Demonstrate the enterprise tokenizer capabilities."""
    
    print("🏢 ENTERPRISE RECIPE TOKENIZER DEMO")
    print("=" * 60)
    
    # Create tokenizer
    enterprise_tokenizer = EnterpriseRecipeTokenizer()
    tokenizer, num_added = enterprise_tokenizer.create_tokenizer()
    
    # Statistics first
    stats = enterprise_tokenizer.get_statistics()
    print(f"\n📈 ENTERPRISE TOKENIZER STATISTICS:")
    print(f"   Total vocabulary: {stats['total_vocabulary_size']:,}")
    print(f"   Special tokens added: {stats['special_tokens_added']:,}")
    print(f"   Culinary categories: {stats['culinary_categories']}")
    
    # Show category breakdown
    print(f"\n📊 VOCABULARY BREAKDOWN:")
    for category, count in stats['category_breakdown'].items():
        print(f"   {category.replace('_', ' ').title()}: {count} terms")
    
    # Demo B2B prompt
    print(f"\n🏢 B2B COMMERCIAL PROMPT DEMO:")
    print("-" * 50)
    
    b2b_prompt = enterprise_tokenizer.create_b2b_prompt(
        "Create a chicken dish with 2 sides for fast-casual service",
        business_type="Fast Casual Restaurant",
        service_style="Quick Service",
        volume="300 servings per day",
        cost_target="Budget-friendly",
        skill_level="Line Cook",
        meal_structure="1 protein + 2 sides",
        servings="4"
    )
    
    print(b2b_prompt[:800] + "...")
    
    # Demo edge case prompt
    print(f"\n🚨 EDGE CASE PROMPT DEMO:")
    print("-" * 50)
    
    edge_prompt = enterprise_tokenizer.create_edge_case_prompt(
        "Create a vegan protein meal under $3 total cost",
        "cost_constraints",
        business_type="School Cafeteria",
        cost_target="Ultra Budget"
    )
    
    print(edge_prompt[:600] + "...")
    
    # Demo expansion prompt
    print(f"\n🌍 EXPANSION PROMPT DEMO:")
    print("-" * 50)
    
    expansion_prompt = enterprise_tokenizer.create_expansion_prompt(
        "Create a signature burger for international franchise",
        "franchise",
        business_type="Fast Casual Chain",
        service_style="Quick Service"
    )
    
    print(expansion_prompt[:600] + "...")
    
    # Tokenization comparison with length compliance
    prompts = {
        "Standard": enterprise_tokenizer.create_enterprise_prompt("Create a chicken recipe"),
        "B2B": b2b_prompt,
        "Edge Case": edge_prompt,
        "Expansion": expansion_prompt
    }
    
    print(f"\n🔍 PROMPT COMPLIANCE ANALYSIS:")
    print("-" * 60)
    print(f"{'Type':12} | {'Chars':>6} | {'Tokens':>6} | {'Ratio':>5} | {'Status':>10}")
    print("-" * 60)
    
    for prompt_type, prompt in prompts.items():
        tokens = tokenizer.encode(prompt)
        ratio = len(prompt)/len(tokens)
        status = "✅ OK" if len(tokens) <= 512 else "⚠️  LONG"
        print(f"{prompt_type:12} | {len(prompt):6,} | {len(tokens):6,} | {ratio:5.1f} | {status:>10}")
    
    # Test truncation on a long prompt
    print(f"\n🔧 TESTING SMART TRUNCATION:")
    print("-" * 40)
    
    long_prompt = enterprise_tokenizer.create_enterprise_prompt(
        "Create a very detailed gourmet restaurant meal with complex preparation techniques and extensive ingredient lists"
    )
    
    original_tokens = len(tokenizer.encode(long_prompt))
    print(f"Original tokens: {original_tokens}")
    
    if original_tokens > 450:
        truncated_prompt = enterprise_tokenizer.smart_truncate_prompt(long_prompt, max_tokens=450)
        truncated_tokens = len(tokenizer.encode(truncated_prompt))
        print(f"Truncated tokens: {truncated_tokens}")
        print(f"Reduction: {original_tokens - truncated_tokens} tokens")
    else:
        print("No truncation needed")
    
    # Show sample B2B tokens
    b2b_tokens = [token for token in enterprise_tokenizer.special_tokens 
                  if any(keyword in token for keyword in ['BUSINESS', 'COMMERCIAL', 'VOLUME', 'COST', 'SCALING'])][:10]
    
    print(f"\n🏢 SAMPLE B2B TOKENS:")
    for token in b2b_tokens:
        print(f"   {token}")
    
    return enterprise_tokenizer

if __name__ == "__main__":
    demo_enterprise_tokenizer()