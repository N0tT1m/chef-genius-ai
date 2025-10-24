#!/usr/bin/env python3
"""
Enterprise B2B Tokenizer for Food Prep Companies
Specialized tokenization for commercial food service, batch cooking, and industrial food preparation
"""

import os
import json
import time
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, decoders, processors, trainers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.processors import BertProcessing
import requests

class EnterpriseB2BTokenizer:
    """
    Comprehensive tokenizer for enterprise food preparation and commercial kitchen operations.
    Optimized for:
    - Large batch cooking (100+ servings)
    - Industrial equipment and processes
    - Food safety and compliance
    - Cost optimization and yield management
    - Supply chain and inventory management
    - Staff training and procedures
    """
    
    def __init__(self, discord_webhook: str = None):
        self.discord_webhook = discord_webhook
        self.vocab_size = 50000  # Target vocab for technical terminology
        self.min_frequency = 1  # Lower min frequency to utilize more vocabulary
        
        # Enterprise-specific vocabularies
        self.enterprise_vocabulary = self._create_enterprise_vocabulary()
        self.special_tokens = self._create_special_tokens()
        
    def _create_enterprise_vocabulary(self) -> Dict[str, List[str]]:
        """Create comprehensive enterprise food service vocabulary."""
        
        return {
            # COMMERCIAL EQUIPMENT & MACHINERY
            "equipment": [
                # Large-scale cooking equipment
                "combi_oven", "convection_steamer", "blast_chiller", "shock_freezer",
                "spiral_mixer", "planetary_mixer", "dough_sheeter", "bread_proofer",
                "salamander", "char_broiler", "flat_top_grill", "induction_range",
                "commercial_fryer", "pressure_fryer", "rotisserie_oven", "pizza_oven",
                "steam_kettle", "tilting_braising_pan", "stock_pot_range", "wok_station",
                
                # Food prep equipment
                "buffalo_chopper", "food_processor_commercial", "vertical_cutter_mixer",
                "deli_slicer", "cheese_grater_commercial", "mandoline_slicer",
                "bone_saw", "meat_grinder_commercial", "vacuum_sealer_chamber",
                "immersion_circulator", "smoking_oven", "dehydrator_commercial",
                
                # Storage and handling
                "walk_in_cooler", "walk_in_freezer", "reach_in_refrigerator",
                "undercounter_refrigerator", "blast_freezer", "holding_cabinet",
                "hot_food_well", "cold_food_well", "steam_table", "sneeze_guard",
                "speed_rail", "ingredient_bin", "cambro_container", "lexan_container"
            ],
            
            # INDUSTRIAL MEASUREMENTS & QUANTITIES
            "measurements": [
                # Volume measurements
                "gallon", "quart", "pint", "fluid_ounce", "liter", "milliliter",
                "case_count", "can_count", "bag_count", "box_count",
                
                # Weight measurements  
                "pound", "ounce", "kilogram", "gram", "ton", "case_weight",
                "net_weight", "gross_weight", "drained_weight", "portion_weight",
                
                # Commercial portions
                "number_10_can", "number_2_can", "hotel_pan", "sixth_pan", "ninth_pan",
                "steam_table_pan", "full_sheet_pan", "half_sheet_pan", "quarter_sheet_pan",
                
                # Yield calculations
                "yield_percentage", "edible_portion", "trim_loss", "cooking_loss",
                "portion_cost", "recipe_cost", "food_cost_percentage", "plate_cost"
            ],
            
            # FOOD SAFETY & COMPLIANCE
            "food_safety": [
                # Temperature control
                "danger_zone", "critical_control_point", "HACCP", "temperature_log",
                "cold_holding", "hot_holding", "rapid_cooling", "two_stage_cooling",
                "thaw_safely", "cook_to_temp", "reheat_to_temp", "hold_at_temp",
                
                # Safety procedures
                "cross_contamination", "allergen_protocol", "gluten_free_prep",
                "kosher_preparation", "halal_preparation", "vegan_prep_area",
                "sanitizer_solution", "test_strips", "wash_rinse_sanitize",
                "hand_washing_station", "glove_change_protocol",
                
                # Documentation
                "temperature_log", "delivery_log", "waste_log", "cleaning_log",
                "training_record", "inspection_report", "corrective_action",
                "supplier_certification", "allergen_documentation"
            ],
            
            # PRODUCTION PLANNING & SCALING
            "production": [
                # Batch cooking
                "batch_size", "production_run", "cook_time_batch", "cooling_time_batch",
                "holding_time", "service_life", "par_level", "reorder_point",
                "prep_schedule", "production_schedule", "cook_schedule",
                
                # Scaling calculations
                "recipe_multiplier", "batch_multiplier", "yield_factor", "conversion_factor",
                "scaling_up", "scaling_down", "portion_control", "waste_factor",
                "shrinkage_allowance", "overproduction_buffer",
                
                # Efficiency metrics
                "labor_hours", "prep_time_per_unit", "cook_time_per_batch",
                "throughput_rate", "capacity_utilization", "equipment_efficiency",
                "energy_consumption", "water_usage", "waste_percentage"
            ],
            
            # COMMERCIAL COOKING TECHNIQUES
            "techniques": [
                # Large-scale methods
                "batch_cooking", "continuous_cooking", "steam_injection_cooking",
                "sous_vide_commercial", "combi_cooking", "pressure_cooking_commercial",
                "blast_chilling", "cook_chill", "cook_freeze", "rethermalization",
                
                # Production techniques
                "mise_en_place_commercial", "prep_station_setup", "line_cooking",
                "expediting", "plating_assembly", "portion_control_plating",
                "speed_service", "buffet_replenishment", "catering_service",
                
                # Quality control
                "taste_testing", "temperature_checking", "visual_inspection",
                "texture_evaluation", "consistency_check", "standardization",
                "recipe_standardization", "procedural_compliance"
            ],
            
            # SUPPLY CHAIN & INVENTORY
            "supply_chain": [
                # Procurement
                "supplier_management", "vendor_qualification", "price_negotiation",
                "contract_pricing", "volume_discounts", "seasonal_pricing",
                "commodity_pricing", "market_fluctuation", "supply_disruption",
                
                # Inventory management
                "inventory_turnover", "FIFO_rotation", "stock_rotation",
                "inventory_tracking", "perpetual_inventory", "cycle_counting",
                "dead_stock", "overstock", "stockout", "safety_stock",
                
                # Receiving and storage
                "receiving_inspection", "quality_check_delivery", "invoice_matching",
                "storage_requirements", "temperature_controlled_storage",
                "dry_storage", "frozen_storage", "refrigerated_storage"
            ],
            
            # COST CONTROL & ANALYTICS
            "cost_control": [
                # Financial metrics
                "food_cost_percentage", "labor_cost_percentage", "total_cost_percentage",
                "gross_profit_margin", "contribution_margin", "break_even_point",
                "cost_per_serving", "cost_per_portion", "recipe_costing",
                
                # Analysis
                "variance_analysis", "cost_deviation", "yield_analysis",
                "waste_analysis", "efficiency_analysis", "profitability_analysis",
                "menu_engineering", "price_optimization", "portion_optimization"
            ],
            
            # STAFF TRAINING & PROCEDURES
            "operations": [
                # Training
                "standard_operating_procedure", "training_protocol", "competency_assessment",
                "skill_certification", "safety_training", "equipment_training",
                "recipe_training", "portion_training", "plating_training",
                
                # Management
                "shift_management", "staff_scheduling", "productivity_tracking",
                "performance_metrics", "quality_standards", "service_standards",
                "cleaning_schedules", "maintenance_schedules", "inspection_schedules"
            ],
            
            # SPECIALIZED DIETARY REQUIREMENTS
            "dietary_commercial": [
                # Institutional dietary
                "therapeutic_diet", "modified_texture", "pureed_diet", "minced_diet",
                "soft_mechanical", "clear_liquid", "full_liquid", "diabetic_diet",
                "low_sodium", "heart_healthy", "renal_diet", "high_protein",
                
                # Large-scale dietary management
                "diet_matrix", "menu_planning_institutional", "nutritional_analysis",
                "calorie_controlled", "portion_controlled", "special_dietary_needs",
                "allergen_free_production", "cross_contamination_prevention"
            ],
            
            # REGULATORY COMPLIANCE
            "compliance": [
                # Health department
                "health_inspection", "permit_renewal", "violation_correction",
                "compliance_audit", "regulatory_update", "code_compliance",
                
                # Industry standards
                "ServSafe_certification", "HACCP_implementation", "GMP_compliance",
                "SQF_certification", "BRC_standard", "organic_certification",
                "FDA_regulation", "USDA_guideline", "local_health_code"
            ]
        }
    
    def _create_special_tokens(self) -> List[str]:
        """Create special tokens for enterprise B2B contexts."""
        
        return [
            # System tokens
            "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
            
            # Enterprise context tokens
            "[BATCH_START]", "[BATCH_END]", "[RECIPE_START]", "[RECIPE_END]",
            "[PROCEDURE_START]", "[PROCEDURE_END]", "[SAFETY_START]", "[SAFETY_END]",
            "[COST_START]", "[COST_END]", "[YIELD_START]", "[YIELD_END]",
            
            # Production tokens
            "[PREP_STATION]", "[COOK_STATION]", "[PLATING_STATION]", "[SERVICE_STATION]",
            "[EQUIPMENT_REQUIRED]", "[TEMPERATURE_CRITICAL]", "[TIME_SENSITIVE]",
            
            # Scale indicators
            "[SMALL_BATCH]", "[MEDIUM_BATCH]", "[LARGE_BATCH]", "[INDUSTRIAL_BATCH]",
            "[SERVING_10]", "[SERVING_50]", "[SERVING_100]", "[SERVING_500]", "[SERVING_1000]",
            
            # Quality indicators
            "[HACCP_POINT]", "[ALLERGEN_ALERT]", "[COST_CRITICAL]", "[EFFICIENCY_FOCUS]",
            "[SKILL_REQUIRED]", "[EQUIPMENT_INTENSIVE]", "[TIME_CRITICAL]",
            
            # Dietary tokens
            "[GLUTEN_FREE]", "[DAIRY_FREE]", "[NUT_FREE]", "[VEGAN]", "[KOSHER]", "[HALAL]",
            "[LOW_SODIUM]", "[DIABETIC]", "[HEART_HEALTHY]", "[HIGH_PROTEIN]",
            
            # Operational tokens
            "[MORNING_PREP]", "[LUNCH_SERVICE]", "[DINNER_SERVICE]", "[CATERING]",
            "[INSTITUTIONAL]", "[HEALTHCARE]", "[EDUCATION]", "[CORPORATE]"
        ]
    
    def create_training_corpus(self, output_file: str = "enterprise_b2b_corpus.txt"):
        """Create comprehensive training corpus for enterprise B2B tokenizer."""
        
        print("🏭 Creating Enterprise B2B Training Corpus...")
        
        corpus_content = []
        
        # Add all vocabulary terms with context
        for category, terms in self.enterprise_vocabulary.items():
            print(f"  📝 Adding {category} vocabulary ({len(terms)} terms)")
            
            for term in terms:
                # Create contextual sentences for each term
                contexts = self._generate_contexts_for_term(term, category)
                corpus_content.extend(contexts)
        
        # Add enterprise-specific recipes and procedures
        enterprise_recipes = self._generate_enterprise_recipes()
        corpus_content.extend(enterprise_recipes)
        
        # Add operational procedures
        procedures = self._generate_operational_procedures()
        corpus_content.extend(procedures)
        
        # Add safety protocols
        safety_protocols = self._generate_safety_protocols()
        corpus_content.extend(safety_protocols)
        
        # Add more diverse training content
        additional_content = self._generate_additional_enterprise_content()
        corpus_content.extend(additional_content)
        
        # Write corpus to file
        with open(output_file, 'w', encoding='utf-8') as f:
            for line in corpus_content:
                f.write(line + '\n')
        
        print(f"✅ Enterprise B2B corpus created: {output_file}")
        print(f"   Total lines: {len(corpus_content):,}")
        print(f"   Estimated tokens: {sum(len(line.split()) for line in corpus_content):,}")
        
        return output_file
    
    def _generate_contexts_for_term(self, term: str, category: str) -> List[str]:
        """Generate contextual sentences for enterprise terms."""
        
        contexts = []
        
        if category == "equipment":
            contexts = [
                f"Set the {term} to 350°F for optimal cooking performance.",
                f"The {term} requires daily cleaning and sanitization procedures.",
                f"Staff training on {term} operation is mandatory before use.",
                f"Monitor {term} temperature logs every 2 hours during service.",
                f"The {term} has a capacity of 40 hotel pans for large batch production."
            ]
        
        elif category == "measurements":
            contexts = [
                f"Recipe calls for 5 {term} of base ingredient for 100 servings.",
                f"Cost analysis shows {term} pricing affects overall food cost by 3%.",
                f"Yield calculation: 1 {term} raw produces 0.75 {term} finished product.",
                f"Storage requirement: maintain {term} inventory at par level.",
                f"Production planning: scale recipe using {term} as base measurement."
            ]
        
        elif category == "food_safety":
            contexts = [
                f"Implement {term} protocol immediately when temperature deviation occurs.",
                f"Document {term} compliance on daily inspection checklist.",
                f"Train all staff on {term} procedures before kitchen assignment.",
                f"The {term} system ensures food safety throughout production.",
                f"Monitor {term} critical control points every 30 minutes."
            ]
        
        elif category == "production":
            contexts = [
                f"Calculate {term} based on service volume and menu requirements.",
                f"The {term} optimization reduces waste by 15% during peak service.",
                f"Schedule {term} activities according to service timeline requirements.",
                f"Monitor {term} efficiency metrics for continuous improvement.",
                f"Adjust {term} parameters based on seasonal demand fluctuations."
            ]
        
        else:
            # Generic contexts for other categories
            contexts = [
                f"The {term} procedure ensures consistent quality in commercial operations.",
                f"Staff must understand {term} requirements for effective implementation.",
                f"Document {term} compliance for regulatory inspection purposes.",
                f"Monitor {term} performance metrics during service periods.",
                f"The {term} system optimizes operational efficiency and cost control."
            ]
        
        return contexts
    
    def _generate_enterprise_recipes(self) -> List[str]:
        """Generate enterprise-scale recipes with commercial specifications."""
        
        recipes = [
            # Large batch main courses
            "[RECIPE_START] [LARGE_BATCH] Commercial Beef Stew - 100 Servings [BATCH_START] Ingredients: 25 lbs beef chuck cut 2-inch cubes, 3 lbs yellow onions diced, 2 lbs carrots diced, 2 lbs celery diced, 1 case number_10_can diced tomatoes, 2 gallons beef stock, 1 cup tomato paste, 2 tbsp worcestershire sauce, bay leaves, thyme, salt, pepper. [PROCEDURE_START] 1. Season beef cubes with salt and pepper 2 hours before cooking. 2. Heat tilting_braising_pan to 400°F. 3. Sear beef in batches until browned on all sides. 4. Add vegetables and cook until softened. 5. Add tomato paste and cook 2 minutes. 6. Deglaze with beef stock. 7. Add remaining ingredients and bring to simmer. 8. Transfer to combi_oven at 325°F for 2.5 hours. [TEMPERATURE_CRITICAL] Internal temperature must reach 165°F. [COST_CRITICAL] Portion cost: $3.25 per 8 oz serving. [YIELD_START] Yield: 100 portions at 8 oz each. [YIELD_END] [PROCEDURE_END] [BATCH_END] [RECIPE_END]",
            
            # Industrial baking
            "[RECIPE_START] [INDUSTRIAL_BATCH] Commercial White Bread - 200 Loaves [BATCH_START] Ingredients: 50 lbs bread flour, 30 lbs water, 2 lbs active dry yeast, 1.5 lbs salt, 1 lb sugar, 0.5 lb shortening. [EQUIPMENT_REQUIRED] spiral_mixer, dough_sheeter, bread_proofer, convection_oven. [PROCEDURE_START] 1. Combine water (80°F) with yeast in spiral_mixer bowl. 2. Add flour and mix on speed 1 for 2 minutes. 3. Add salt, sugar, and shortening. 4. Mix on speed 2 for 8 minutes until gluten development. 5. Bulk ferment 1 hour at 78°F. 6. Divide into 2 lb portions using dough_sheeter. 7. Shape into loaves and place in pans. 8. Proof in bread_proofer at 85°F, 85% humidity for 45 minutes. 9. Bake at 375°F for 35 minutes. [TEMPERATURE_CRITICAL] Internal temperature 190°F. [HACCP_POINT] Monitor proofing temperature and humidity. [YIELD_START] Yield: 200 loaves at 2 lbs each. [YIELD_END] [PROCEDURE_END] [BATCH_END] [RECIPE_END]",
            
            # Food service preparation
            "[RECIPE_START] [MEDIUM_BATCH] Institutional Chicken Salad - 50 Servings [BATCH_START] [HEALTHCARE] dietary requirements compliant. Ingredients: 12 lbs cooked chicken breast diced, 2 lbs celery diced, 1 lb mayonnaise, 0.5 lb yellow onion minced, 2 tbsp lemon juice, salt, white pepper. [ALLERGEN_ALERT] Contains eggs (mayonnaise). [PROCEDURE_START] 1. Cook chicken breast in combi_oven at 325°F to 165°F internal temperature. 2. Cool rapidly in blast_chiller to 40°F within 4 hours. 3. Dice chicken into 0.5-inch pieces. 4. Combine mayonnaise, lemon juice, and seasonings. 5. Fold chicken, celery, and onion into dressing. 6. Portion into 4 oz servings using portion_control scoops. [TEMPERATURE_CRITICAL] Maintain below 40°F during preparation. [SAFETY_START] Wash hands and change gloves between tasks. [SAFETY_END] [COST_CRITICAL] Portion cost: $2.15 per 4 oz serving. [YIELD_START] Yield: 50 servings at 4 oz each. [YIELD_END] [PROCEDURE_END] [BATCH_END] [RECIPE_END]"
        ]
        
        return recipes
    
    def _generate_operational_procedures(self) -> List[str]:
        """Generate operational procedures for commercial kitchens."""
        
        procedures = [
            "[PROCEDURE_START] Opening Kitchen Checklist [MORNING_PREP] 1. Check walk_in_cooler and walk_in_freezer temperatures - record on temperature_log. 2. Inspect all equipment for cleanliness and proper function. 3. Review prep_schedule and production_schedule for the day. 4. Set up prep_stations with required tools and mise_en_place. 5. Check par_levels for all ingredients and supplies. 6. Verify staff_scheduling matches service requirements. 7. Review special_dietary_needs for today's service. [HACCP_POINT] All temperatures must be within safe ranges before food production begins. [PROCEDURE_END]",
            
            "[PROCEDURE_START] Batch Cooking Protocol [LARGE_BATCH] 1. Calculate batch_size based on service projections and par_levels. 2. Verify recipe_multiplier calculations for scaling. 3. Pre-heat equipment to specified temperatures. 4. Gather all ingredients using mise_en_place principles. 5. Follow standard_operating_procedure for recipe execution. 6. Monitor critical_control_points throughout cooking process. 7. Test final product for taste, texture, and temperature. 8. Transfer to appropriate holding_equipment at correct temperatures. 9. Label with production time, batch size, and use-by date. 10. Document batch_information for inventory_tracking. [EFFICIENCY_FOCUS] Target throughput_rate: 100 servings per labor hour. [PROCEDURE_END]",
            
            "[PROCEDURE_START] Equipment Cleaning Protocol [EQUIPMENT_INTENSIVE] 1. Allow equipment to cool to safe handling temperature. 2. Disconnect power and gas connections per safety_protocol. 3. Remove all food debris and loose particles. 4. Pre-rinse with warm water to remove grease and buildup. 5. Apply approved sanitizer_solution at correct concentration. 6. Scrub all surfaces with appropriate cleaning tools. 7. Rinse thoroughly with clean water. 8. Apply final sanitizer_solution and air dry. 9. Reassemble equipment according to manufacturer specifications. 10. Test equipment function before returning to service. [SAFETY_START] Wear protective equipment and follow lockout_tagout procedures. [SAFETY_END] [PROCEDURE_END]"
        ]
        
        return procedures
    
    def _generate_safety_protocols(self) -> List[str]:
        """Generate food safety protocols for commercial operations."""
        
        protocols = [
            "[SAFETY_START] Temperature Control Protocol [TEMPERATURE_CRITICAL] 1. Monitor all cold_storage units every 2 hours during service. 2. Record temperatures on temperature_log with staff initials. 3. Investigate any temperature deviation immediately. 4. Implement corrective_action if temperatures exceed safe ranges. 5. For cold_holding: maintain 40°F or below at all times. 6. For hot_holding: maintain 140°F or above during service. 7. Use calibrated thermometers for all temperature measurements. 8. Document any equipment malfunctions or temperature excursions. [HACCP_POINT] Critical control point requires immediate action if temperatures deviate. [SAFETY_END]",
            
            "[SAFETY_START] Allergen Management Protocol [ALLERGEN_ALERT] 1. Maintain separate prep_areas for allergen_free_production. 2. Use dedicated cutting boards, utensils, and equipment. 3. Clean and sanitize all surfaces before allergen_free preparation. 4. Train all staff on cross_contamination prevention procedures. 5. Label all containers with allergen information clearly. 6. Verify ingredient labels for hidden allergens before use. 7. Communicate allergen presence to service staff clearly. 8. Document allergen_protocol compliance on daily logs. [SKILL_REQUIRED] All staff must complete allergen_training certification. [SAFETY_END]",
            
            "[SAFETY_START] Personal Hygiene Protocol 1. Wash hands for 20 seconds with soap and warm water. 2. Change gloves between tasks and every 4 hours minimum. 3. Use hand_washing_station properly - soap, wash, rinse, sanitize. 4. Wear clean uniform and hair restraint during all shifts. 5. Report illness to management before working with food. 6. No jewelry except plain wedding band permitted in prep areas. 7. Cover all cuts and wounds with waterproof bandages. 8. Follow glove_change_protocol when switching between tasks. [COMPLIANCE] Health_department requires strict adherence to hygiene standards. [SAFETY_END]"
        ]
        
        return protocols
    
    def _generate_additional_enterprise_content(self) -> List[str]:
        """Generate additional diverse enterprise training content."""
        
        content = []
        
        # Generate batch recipe variations
        for i in range(50):
            content.append(f"[RECIPE_START] [LARGE_BATCH] Commercial Recipe {i+1}: Set combi_oven to 325°F and prepare 100 servings using standard_operating_procedure. Calculate recipe_multiplier for batch_size optimization. Monitor temperature_log every 30 minutes for HACCP compliance. Document yield_percentage and food_cost_percentage for cost_control analysis. [RECIPE_END]")
        
        # Generate equipment procedures
        equipment_list = [
            "combi_oven", "blast_chiller", "tilting_braising_pan", "spiral_mixer", "walk_in_cooler",
            "steam_kettle", "buffalo_chopper", "vacuum_sealer_chamber", "salamander", "convection_steamer"
        ]
        
        for equipment in equipment_list:
            for i in range(10):
                content.append(f"[PROCEDURE_START] {equipment} Operation Protocol: Follow standard_operating_procedure for {equipment} startup and operation. Monitor temperature_log and equipment_efficiency metrics. Implement HACCP critical_control_points throughout operation. Document maintenance_schedule and cleaning_protocol compliance. [PROCEDURE_END]")
        
        # Generate food safety scenarios
        for i in range(30):
            content.append(f"[SAFETY_START] Food Safety Protocol {i+1}: Implement HACCP critical_control_point monitoring for temperature_critical operations. Document corrective_action procedures and compliance_audit findings. Train staff on cross_contamination prevention and allergen_protocol implementation. [SAFETY_END]")
        
        # Generate cost control scenarios
        for i in range(30):
            content.append(f"[COST_START] Cost Control Analysis {i+1}: Calculate food_cost_percentage and yield_percentage for recipe_multiplier optimization. Monitor portion_cost and recipe_cost for variance_analysis. Implement inventory_turnover and supplier_management strategies for cost_optimization. [COST_END]")
        
        # Generate production planning content
        production_scenarios = [
            "batch_cooking", "production_schedule", "prep_schedule", "scaling_up", "scaling_down",
            "par_level", "reorder_point", "throughput_rate", "capacity_utilization", "equipment_efficiency"
        ]
        
        for scenario in production_scenarios:
            for i in range(15):
                content.append(f"Production planning for {scenario} requires careful calculation of recipe_multiplier and batch_size optimization. Monitor equipment_efficiency and throughput_rate for capacity_utilization analysis. Implement standard_operating_procedure for consistent results and cost_control.")
        
        # Generate supply chain content
        for i in range(25):
            content.append(f"Supply chain management involves supplier_qualification, inventory_tracking, and FIFO_rotation procedures. Monitor price_negotiation opportunities and volume_discounts for cost_optimization. Implement receiving_inspection and quality_check_delivery protocols for compliance.")
        
        return content
    
    def train_tokenizer(self, corpus_file: str) -> PreTrainedTokenizerFast:
        """Train the enterprise B2B tokenizer on the corpus."""
        
        print("🔧 Training Enterprise B2B Tokenizer...")
        
        # Initialize tokenizer components
        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        
        # Set normalizer
        tokenizer.normalizer = normalizers.Sequence([
            NFD(),
            Lowercase(),
            StripAccents()
        ])
        
        # Set pre-tokenizer
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
            Whitespace(),
            Punctuation()
        ])
        
        # Post-processor will be set after training
        
        # Set decoder
        tokenizer.decoder = decoders.BPEDecoder(suffix="</w>")
        
        # Create trainer with adjusted parameters for larger corpus
        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=1,  # Lower min frequency to utilize more vocabulary
            special_tokens=self.special_tokens,
            show_progress=True,
            continuing_subword_prefix="##"
        )
        
        # Train on corpus
        print(f"  📚 Training on corpus: {corpus_file}")
        tokenizer.train([corpus_file], trainer)
        
        # Set post-processor after training
        try:
            sep_token_id = tokenizer.token_to_id("[SEP]")
            cls_token_id = tokenizer.token_to_id("[CLS]")
            if sep_token_id is not None and cls_token_id is not None:
                from tokenizers.processors import BertProcessing
                tokenizer.post_processor = BertProcessing(
                    ("[SEP]", sep_token_id),
                    ("[CLS]", cls_token_id)
                )
        except Exception as e:
            print(f"⚠️ Could not set post-processor: {e}")
        
        # Convert to HuggingFace tokenizer
        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token="[UNK]",
            pad_token="[PAD]",
            cls_token="[CLS]",
            sep_token="[SEP]",
            mask_token="[MASK]"
        )
        
        print("✅ Enterprise B2B Tokenizer training complete!")
        print(f"   Vocabulary size: {len(hf_tokenizer.get_vocab()):,}")
        print(f"   Special tokens: {len(self.special_tokens)}")
        
        return hf_tokenizer
    
    def save_tokenizer(self, tokenizer: PreTrainedTokenizerFast, output_dir: str = "enterprise_b2b_tokenizer"):
        """Save the trained tokenizer."""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Save tokenizer
        tokenizer.save_pretrained(output_dir)
        
        # Save configuration
        config = {
            "tokenizer_type": "enterprise_b2b",
            "vocab_size": self.vocab_size,
            "min_frequency": self.min_frequency,
            "special_tokens": self.special_tokens,
            "enterprise_categories": list(self.enterprise_vocabulary.keys()),
            "total_enterprise_terms": sum(len(terms) for terms in self.enterprise_vocabulary.values()),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_domain": "commercial_food_service"
        }
        
        with open(f"{output_dir}/enterprise_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Enterprise B2B Tokenizer saved to: {output_dir}")
        return output_dir
    
    def test_tokenizer(self, tokenizer: PreTrainedTokenizerFast) -> Dict[str, Any]:
        """Test the enterprise tokenizer with commercial food service examples."""
        
        print("🧪 Testing Enterprise B2B Tokenizer...")
        
        test_texts = [
            "Set the combi_oven to 325°F and prepare 100 servings of beef stew using the tilting_braising_pan for initial searing.",
            "Monitor the blast_chiller temperature every 30 minutes and document on the temperature_log for HACCP compliance.",
            "Calculate the recipe_multiplier for scaling from 50 to 200 servings while maintaining portion_cost targets.",
            "Implement allergen_protocol for gluten_free preparation using dedicated equipment and separate prep_areas.",
            "The walk_in_cooler temperature exceeded 40°F requiring immediate corrective_action and waste_assessment.",
            "Train staff on standard_operating_procedure for the spiral_mixer and dough_sheeter operations.",
            "Review food_cost_percentage analysis showing 3% variance from budget targets this quarter.",
            "Prepare therapeutic_diet modifications for healthcare facility requiring low_sodium and diabetic_diet options."
        ]
        
        results = []
        total_tokens = 0
        
        for i, text in enumerate(test_texts, 1):
            print(f"  Test {i}: {text[:60]}...")
            
            # Tokenize
            tokens = tokenizer.tokenize(text)
            token_ids = tokenizer.encode(text)
            decoded = tokenizer.decode(token_ids)
            
            # Count enterprise terms
            enterprise_terms_found = []
            for category, terms in self.enterprise_vocabulary.items():
                for term in terms:
                    if term in text.lower():
                        enterprise_terms_found.append(term)
            
            result = {
                "text": text,
                "token_count": len(tokens),
                "enterprise_terms_found": enterprise_terms_found,
                "tokens_sample": tokens[:10],
                "decoded_match": decoded == text
            }
            
            results.append(result)
            total_tokens += len(tokens)
        
        # Calculate statistics
        avg_tokens_per_text = total_tokens / len(test_texts)
        enterprise_term_coverage = sum(len(r["enterprise_terms_found"]) for r in results)
        
        test_summary = {
            "total_tests": len(test_texts),
            "total_tokens": total_tokens,
            "avg_tokens_per_text": avg_tokens_per_text,
            "enterprise_terms_identified": enterprise_term_coverage,
            "successful_round_trips": sum(1 for r in results if r["decoded_match"]),
            "results": results
        }
        
        print(f"✅ Enterprise B2B Tokenizer Testing Complete!")
        print(f"   Average tokens per text: {avg_tokens_per_text:.1f}")
        print(f"   Enterprise terms identified: {enterprise_term_coverage}")
        print(f"   Successful round trips: {test_summary['successful_round_trips']}/{len(test_texts)}")
        
        return test_summary
    
    def send_discord_notification(self, tokenizer_path: str, test_results: Dict[str, Any]):
        """Send training completion notification to Discord."""
        
        if not self.discord_webhook:
            return
        
        try:
            embed = {
                "title": "🏭 Enterprise B2B Tokenizer Training Complete",
                "description": "Specialized tokenizer for commercial food service operations",
                "color": 0x0066cc,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "fields": [
                    {
                        "name": "📊 Tokenizer Stats",
                        "value": f"🔤 Vocabulary: {self.vocab_size:,} tokens\n🏷️ Special tokens: {len(self.special_tokens)}\n📁 Saved to: `{tokenizer_path}`",
                        "inline": True
                    },
                    {
                        "name": "🧪 Test Results",
                        "value": f"✅ Tests passed: {test_results['successful_round_trips']}/{test_results['total_tests']}\n📊 Avg tokens/text: {test_results['avg_tokens_per_text']:.1f}\n🏭 Enterprise terms: {test_results['enterprise_terms_identified']}",
                        "inline": True
                    },
                    {
                        "name": "🎯 Specialized For",
                        "value": "• Commercial kitchen operations\n• Large batch cooking (100+ servings)\n• Food safety compliance\n• Supply chain management\n• Cost control & analytics",
                        "inline": False
                    },
                    {
                        "name": "📋 Enterprise Categories",
                        "value": f"Equipment • Safety • Production • Cost Control • Compliance • Supply Chain • Operations",
                        "inline": False
                    }
                ],
                "footer": {"text": "Enterprise B2B Tokenizer • Chef Genius"}
            }
            
            payload = {
                "embeds": [embed],
                "username": "Chef Genius Tokenizer Bot"
            }
            
            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
            print("🔔 Discord notification sent successfully!")
            
        except Exception as e:
            print(f"⚠️ Failed to send Discord notification: {e}")

def main():
    """Main function for creating and training enterprise B2B tokenizer."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Enterprise B2B Tokenizer for Commercial Food Service')
    parser.add_argument('--output-dir', type=str, default='enterprise_b2b_tokenizer', help='Output directory for tokenizer')
    parser.add_argument('--vocab-size', type=int, default=50000, help='Vocabulary size')
    parser.add_argument('--discord-webhook', type=str, 
                       default='https://discord.com/api/webhooks/1386109570283343953/uGkhj9dpuCg09SbKzZ0Tx2evugJrchQv-nrq3w0r_xi3w8si-XBpQJuxq_p_bcQlhB9W',
                       help='Discord webhook for notifications')
    
    args = parser.parse_args()
    
    print("🏭 ENTERPRISE B2B TOKENIZER TRAINING")
    print("Specialized for commercial food service operations")
    print("=" * 80)
    
    # Create tokenizer
    tokenizer_trainer = EnterpriseB2BTokenizer(discord_webhook=args.discord_webhook)
    tokenizer_trainer.vocab_size = args.vocab_size
    
    # Create training corpus
    corpus_file = tokenizer_trainer.create_training_corpus()
    
    # Train tokenizer
    trained_tokenizer = tokenizer_trainer.train_tokenizer(corpus_file)
    
    # Test tokenizer
    test_results = tokenizer_trainer.test_tokenizer(trained_tokenizer)
    
    # Save tokenizer
    saved_path = tokenizer_trainer.save_tokenizer(trained_tokenizer, args.output_dir)
    
    # Send notification
    tokenizer_trainer.send_discord_notification(saved_path, test_results)
    
    print(f"\n🎉 Enterprise B2B Tokenizer Complete!")
    print(f"📁 Saved to: {saved_path}")
    print(f"🧪 Test success rate: {test_results['successful_round_trips']}/{test_results['total_tests']}")

if __name__ == "__main__":
    main()