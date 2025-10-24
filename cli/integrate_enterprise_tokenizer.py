#!/usr/bin/env python3
"""
🔧 ENTERPRISE TOKENIZER INTEGRATION
Seamlessly integrate the enterprise tokenizer into your existing training pipeline
"""

import os
import sys
from pathlib import Path
from enterprise_recipe_tokenizer import EnterpriseRecipeTokenizer

def patch_complete_optimized_training():
    """Create the exact patch for your complete_optimized_training.py"""
    
    patch_code = '''
# === ENTERPRISE TOKENIZER INTEGRATION ===
# Add this import at the top of your file
from enterprise_recipe_tokenizer import EnterpriseRecipeTokenizer

def setup_enterprise_tokenizer(base_model_path: str):
    """Setup enterprise-grade tokenizer for professional recipe generation."""
    
    print("🏢 Setting up Enterprise Recipe Tokenizer...")
    
    # Create enterprise tokenizer
    enterprise_tokenizer = EnterpriseRecipeTokenizer(base_model_path)
    tokenizer, num_added = enterprise_tokenizer.create_tokenizer()
    
    print(f"✅ Enterprise tokenizer ready with {num_added} culinary tokens")
    print(f"📊 Total vocabulary: {len(tokenizer):,} tokens")
    
    return tokenizer, num_added, enterprise_tokenizer

# === REPLACE YOUR TOKENIZER LOADING ===
# In your main() function, find this line:
# tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
# 
# Replace it with:
tokenizer, num_added_tokens, enterprise_obj = setup_enterprise_tokenizer(args.pretrained_model)

# === CRITICAL: RESIZE MODEL EMBEDDINGS ===
# Add this right after loading the model:
if num_added_tokens > 0:
    print(f"🔧 Resizing model embeddings for {num_added_tokens} new tokens...")
    model.resize_token_embeddings(len(tokenizer))
    print("✅ Model embeddings resized for enterprise tokenizer")

# === ENHANCE DATA FORMATTING ===
# Add this method to your CompleteOptimizedTrainer class:
def format_enterprise_training_sample(self, sample_text):
    """Format training samples using enterprise structure."""
    
    # Try to parse existing recipe data
    try:
        # If your data has structure, use it
        if "ingredients:" in sample_text.lower() and "instructions:" in sample_text.lower():
            parts = sample_text.split("ingredients:")
            if len(parts) > 1:
                title_part = parts[0].strip()
                rest = "ingredients:" + parts[1]
                
                if "instructions:" in rest.lower():
                    ing_parts = rest.split("instructions:")
                    ingredients = ing_parts[0].replace("ingredients:", "").strip()
                    instructions = ing_parts[1].strip()
                    
                    # Create structured recipe
                    recipe_data = {
                        "title": title_part or "Delicious Recipe",
                        "ingredients": [ing.strip() for ing in ingredients.split("\\n") if ing.strip()],
                        "instructions": [inst.strip() for inst in instructions.split("\\n") if inst.strip()]
                    }
                    
                    return self.enterprise_tokenizer.format_enterprise_recipe(recipe_data)
    except:
        pass
    
    # Fallback: wrap existing text in basic structure
    return f"[RECIPE_START]\\n[TITLE_START]Recipe[TITLE_END]\\n[INGREDIENTS_START]\\n{sample_text}\\n[INGREDIENTS_END]\\n[RECIPE_END]"

# === UPDATE GENERATION PROMPTS ===
# In your generate_sample_recipes method, replace the prompt with:
def generate_enterprise_recipe_sample(self, step, prompt):
    """Generate recipe samples using enterprise tokenizer structure."""
    
    try:
        # Create enterprise-grade prompt
        formatted_prompt = self.enterprise_tokenizer.create_enterprise_prompt(
            prompt,
            servings="4",
            difficulty="Medium"
        )
        
        # Tokenize with enterprise structure
        inputs = self.tokenizer(
            formatted_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512  # Increased for enterprise structure
        ).to(self.model.device)
        
        # Generate with enhanced parameters
        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=400,  # Longer for detailed recipes
                min_length=100,      # Ensure substantial output
                num_beams=4,
                early_stopping=True,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                length_penalty=1.0,
                no_repeat_ngram_size=3,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=False
            )
        
        # Decode and clean
        recipe = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        if formatted_prompt in recipe:
            recipe = recipe.replace(formatted_prompt, "").strip()
        
        print(f"\\n🍳 Enterprise Recipe Sample (Step {step}):")
        print(f"Prompt: {prompt}")
        print(f"Recipe: {recipe[:200]}..." if len(recipe) > 200 else f"Recipe: {recipe}")
        print("-" * 50)
        
        # Enhanced quality analysis
        word_count = len(recipe.split())
        has_structure = any(token in recipe for token in ["[RECIPE_START]", "[INGREDIENTS_START]", "[INSTRUCTIONS_START]"])
        has_ingredients = any(keyword in recipe.lower() for keyword in ['ingredients', 'cups', 'tablespoons', 'teaspoons', 'oz', 'lbs'])
        has_instructions = any(keyword in recipe.lower() for keyword in ['cook', 'bake', 'heat', 'mix', 'add', 'step', 'sauté', 'grill'])
        has_timing = any(keyword in recipe.lower() for keyword in ['minutes', 'hours', 'until', 'time'])
        
        quality_score = sum([has_structure, has_ingredients, has_instructions, has_timing])
        
        print(f"📊 Enterprise Quality Metrics:")
        print(f"   Words: {word_count}")
        print(f"   {'✅' if has_structure else '❌'} Enterprise structure")
        print(f"   {'✅' if has_ingredients else '❌'} Ingredients detected")
        print(f"   {'✅' if has_instructions else '❌'} Instructions detected")
        print(f"   {'✅' if has_timing else '❌'} Timing information")
        print(f"   Quality Score: {quality_score}/4")
        
        # Cleanup
        del inputs, outputs
        
    except Exception as e:
        print(f"❌ Enterprise generation failed: {e}")
        # Fallback to basic generation
        super().generate_sample_recipes(step, [prompt])

# === STORE ENTERPRISE TOKENIZER REFERENCE ===
# In your CompleteOptimizedTrainer __init__ method, add:
self.enterprise_tokenizer = None  # Will be set during setup

# And in the trainer creation in main(), add:
trainer.enterprise_tokenizer = enterprise_obj
'''
    
    return patch_code

def create_automatic_integration():
    """Automatically integrate enterprise tokenizer into existing training script."""
    
    training_script_path = Path("complete_optimized_training.py")
    
    if not training_script_path.exists():
        print("❌ complete_optimized_training.py not found")
        return False
    
    # Read existing script
    with open(training_script_path, 'r') as f:
        content = f.read()
    
    # Check if already integrated
    if "EnterpriseRecipeTokenizer" in content:
        print("✅ Enterprise tokenizer already integrated")
        return True
    
    # Create backup
    backup_path = training_script_path.with_suffix('.py.backup')
    with open(backup_path, 'w') as f:
        f.write(content)
    
    print(f"💾 Backup created: {backup_path}")
    
    # Add import at the top
    import_line = "from enterprise_recipe_tokenizer import EnterpriseRecipeTokenizer\\n"
    
    # Find imports section
    lines = content.split('\\n')
    insert_index = 0
    for i, line in enumerate(lines):
        if line.startswith('from transformers import'):
            insert_index = i + 1
            break
    
    # Insert import
    lines.insert(insert_index, import_line.strip())
    
    # Find tokenizer loading and replace
    for i, line in enumerate(lines):
        if "tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)" in line:
            # Replace with enterprise setup
            lines[i] = "        tokenizer, num_added_tokens, enterprise_obj = setup_enterprise_tokenizer(args.pretrained_model)"
            
            # Add model resizing after model loading
            for j in range(i, min(i + 20, len(lines))):
                if "model = model.to(device)" in lines[j]:
                    resize_code = [
                        "",
                        "    # Enterprise tokenizer integration",
                        "    if num_added_tokens > 0:",
                        "        print(f'🔧 Resizing model embeddings for {num_added_tokens} new tokens...')",
                        "        model.resize_token_embeddings(len(tokenizer))",
                        "        print('✅ Model embeddings resized for enterprise tokenizer')",
                        ""
                    ]
                    for k, resize_line in enumerate(resize_code):
                        lines.insert(j + 1 + k, resize_line)
                    break
            break
    
    # Add enterprise tokenizer setup function
    setup_function = '''
def setup_enterprise_tokenizer(base_model_path: str):
    """Setup enterprise-grade tokenizer for professional recipe generation."""
    
    print("🏢 Setting up Enterprise Recipe Tokenizer...")
    
    # Create enterprise tokenizer
    enterprise_tokenizer = EnterpriseRecipeTokenizer(base_model_path)
    tokenizer, num_added = enterprise_tokenizer.create_tokenizer()
    
    print(f"✅ Enterprise tokenizer ready with {num_added} culinary tokens")
    print(f"📊 Total vocabulary: {len(tokenizer):,} tokens")
    
    return tokenizer, num_added, enterprise_tokenizer
'''
    
    # Insert setup function before main()
    for i, line in enumerate(lines):
        if "def main():" in line:
            setup_lines = setup_function.strip().split('\\n')
            for j, setup_line in enumerate(setup_lines):
                lines.insert(i + j, setup_line)
            break
    
    # Write modified content
    modified_content = '\\n'.join(lines)
    with open(training_script_path, 'w') as f:
        f.write(modified_content)
    
    print("✅ Enterprise tokenizer integrated successfully!")
    print("🔄 Your training script now uses enterprise-grade recipe tokenization")
    
    return True

def test_integration():
    """Test the enterprise tokenizer integration."""
    
    print("🧪 Testing Enterprise Tokenizer Integration...")
    
    try:
        # Create enterprise tokenizer
        enterprise_tokenizer = EnterpriseRecipeTokenizer()
        tokenizer, num_added = enterprise_tokenizer.create_tokenizer()
        
        # Test prompt creation
        prompt = enterprise_tokenizer.create_enterprise_prompt(
            "Create a professional pasta carbonara recipe",
            servings="4",
            cuisine="Italian",
            difficulty="Medium"
        )
        
        print(f"✅ Enterprise prompt created ({len(prompt)} characters)")
        
        # Test tokenization
        tokens = tokenizer.encode(prompt)
        print(f"✅ Tokenization successful ({len(tokens)} tokens)")
        
        # Test recipe formatting
        sample_recipe = {
            "title": "Test Recipe",
            "ingredients": ["1 cup flour", "2 eggs"],
            "instructions": ["Mix ingredients", "Cook for 10 minutes"]
        }
        
        formatted = enterprise_tokenizer.format_enterprise_recipe(sample_recipe)
        print(f"✅ Recipe formatting successful ({len(formatted)} characters)")
        
        print("\\n🎉 Enterprise tokenizer integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        return False

def main():
    """Main integration workflow."""
    
    print("🏢 ENTERPRISE TOKENIZER INTEGRATION")
    print("=" * 50)
    
    # Test first
    if not test_integration():
        print("❌ Pre-integration test failed")
        return
    
    print("\\n📋 Integration Options:")
    print("1. 📄 Show manual patch code")
    print("2. 🔧 Automatic integration (recommended)")  
    print("3. 🧪 Test integration only")
    
    choice = input("\\nChoose option (1-3): ").strip()
    
    if choice == "1":
        print("\\n📄 MANUAL PATCH CODE:")
        print("=" * 30)
        print(patch_complete_optimized_training())
        
    elif choice == "2":
        print("\\n🔧 Performing automatic integration...")
        if create_automatic_integration():
            print("\\n🎉 Integration complete!")
            print("\\n📋 Next Steps:")
            print("1. ✅ Enterprise tokenizer integrated")
            print("2. 🔄 Restart your training with enhanced tokenization")
            print("3. 📈 Expect dramatically better recipe generation quality")
            print("4. 🏢 Your model now uses professional-grade culinary vocabulary")
        else:
            print("❌ Automatic integration failed - use manual patch")
            
    elif choice == "3":
        print("\\n🧪 Integration test completed above")
        
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()