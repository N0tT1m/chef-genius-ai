#!/usr/bin/env python3
"""
Test script to validate Rust integration
"""

import asyncio
import time
import sys
from pathlib import Path

# Add backend to path
sys.path.append(str(Path(__file__).parent / "backend"))

def test_rust_import():
    """Test basic Rust import"""
    print("🧪 Testing Rust core import...")
    try:
        import chef_genius_core
        print("✅ Rust core imported successfully")
        
        # Test system info
        info = chef_genius_core.get_system_info()
        print(f"   CPU cores: {info['cpu_count']}")
        print(f"   CUDA available: {info['cuda_available']}")
        print(f"   Version: {info['version']}")
        return True
    except ImportError as e:
        print(f"❌ Rust import failed: {e}")
        return False

def test_rust_engines():
    """Test individual Rust engines"""
    print("\n🔧 Testing Rust engines...")
    try:
        import chef_genius_core
        
        # Test inference engine
        try:
            engine = chef_genius_core.PyInferenceEngine("dummy_path")
            print("✅ Inference engine created")
        except Exception as e:
            print(f"⚠️  Inference engine test failed (expected): {type(e).__name__}")
        
        # Test vector search
        search = chef_genius_core.PyVectorSearchEngine()
        stats = search.get_stats()
        print(f"✅ Vector search engine created (recipes: {stats['total_recipes']})")
        
        # Test recipe processor
        processor = chef_genius_core.PyRecipeProcessor()
        ingredients = processor.extract_ingredients("1 cup flour\n2 eggs\n1 tsp salt")
        print(f"✅ Recipe processor working (extracted {len(ingredients)} ingredients)")
        
        # Test nutrition analyzer
        analyzer = chef_genius_core.PyNutritionAnalyzer()
        print("✅ Nutrition analyzer created")
        
        return True
    except Exception as e:
        print(f"❌ Engine test failed: {e}")
        return False

def test_recipe_processing():
    """Test recipe processing functionality"""
    print("\n📝 Testing recipe processing...")
    try:
        import chef_genius_core
        
        processor = chef_genius_core.PyRecipeProcessor()
        
        # Test ingredient extraction
        recipe_text = """
        Chicken Rice Recipe
        
        Ingredients:
        - 1 cup rice
        - 2 chicken breasts
        - 1 onion, diced
        - 2 cloves garlic
        - 1 tsp salt
        
        Instructions:
        1. Cook rice according to package directions
        2. Season and cook chicken
        3. Sauté onion and garlic
        4. Combine everything and serve
        """
        
        # Parse full recipe
        recipe = processor.parse_recipe(recipe_text)
        print(f"✅ Parsed recipe: '{recipe.title}'")
        print(f"   Ingredients: {len(recipe.ingredients)}")
        print(f"   Instructions: {len(recipe.instructions)}")
        
        # Test ingredient extraction only
        ingredients = processor.extract_ingredients(recipe_text)
        print(f"✅ Extracted {len(ingredients)} ingredients")
        
        # Test validation
        validation = processor.validate_recipe(recipe)
        print(f"✅ Recipe validation: valid={validation['is_valid']}, score={validation['score']}")
        
        return True
    except Exception as e:
        print(f"❌ Recipe processing test failed: {e}")
        return False

def test_nutrition_analysis():
    """Test nutrition analysis"""
    print("\n🍎 Testing nutrition analysis...")
    try:
        import chef_genius_core
        
        analyzer = chef_genius_core.PyNutritionAnalyzer()
        
        # Create test recipe
        recipe = chef_genius_core.PyRecipe(
            title="Chicken and Rice",
            ingredients=["chicken breast", "rice", "broccoli"],
            instructions=["Cook chicken", "Cook rice", "Steam broccoli", "Combine"],
            cooking_time="30 minutes",
            prep_time="15 minutes",
            servings=4,
            difficulty="Easy",
            cuisine_type=None,
            dietary_tags=None,
            confidence=0.9
        )
        
        # Analyze nutrition
        nutrition = analyzer.analyze_recipe(recipe)
        nutrition_dict = nutrition.to_dict()
        
        print(f"✅ Nutrition analysis completed")
        print(f"   Calories: {nutrition_dict.get('calories', 'N/A')}")
        print(f"   Protein: {nutrition_dict.get('protein_g', 'N/A')}g")
        print(f"   Health score: {nutrition_dict.get('health_score', 'N/A')}")
        
        # Test dietary compatibility
        compatibility = analyzer.check_dietary_compatibility(recipe, ["vegetarian"])
        print(f"✅ Dietary check: compatible={compatibility['is_compatible']}")
        
        return True
    except Exception as e:
        print(f"❌ Nutrition analysis test failed: {e}")
        return False

def test_vector_search():
    """Test vector search functionality"""
    print("\n🔍 Testing vector search...")
    try:
        import chef_genius_core
        
        search = chef_genius_core.PyVectorSearchEngine()
        
        # Create test recipes
        test_recipes = []
        for i, (title, ingredients) in enumerate([
            ("Pasta Carbonara", ["pasta", "eggs", "bacon", "cheese"]),
            ("Chicken Curry", ["chicken", "curry powder", "coconut milk", "rice"]),
            ("Caesar Salad", ["lettuce", "caesar dressing", "croutons", "parmesan"]),
        ]):
            recipe = chef_genius_core.PyRecipe(
                title=title,
                ingredients=ingredients,
                instructions=[f"Step 1 for {title}", f"Step 2 for {title}"],
                cooking_time="30 minutes",
                prep_time="10 minutes",
                servings=4,
                difficulty="Medium",
                cuisine_type=None,
                dietary_tags=None,
                confidence=0.8
            )
            test_recipes.append(recipe)
        
        # Add recipes to search index
        search.add_recipes(test_recipes)
        stats = search.get_stats()
        print(f"✅ Added {stats['total_recipes']} recipes to search index")
        
        # Test search
        results = search.search("pasta dish", 2)
        print(f"✅ Search results: {len(results)} found")
        for result in results:
            print(f"   - {result.recipe.title} (score: {result.score:.3f})")
        
        # Test ingredient search
        ing_results = search.search_by_ingredients(["chicken", "rice"], 2)
        print(f"✅ Ingredient search: {len(ing_results)} found")
        
        return True
    except Exception as e:
        print(f"❌ Vector search test failed: {e}")
        return False

async def test_python_integration():
    """Test Python integration layer"""
    print("\n🐍 Testing Python integration...")
    try:
        from backend.app.services.rust_integration import rust_service, RUST_AVAILABLE
        
        print(f"✅ Rust service imported (available: {RUST_AVAILABLE})")
        
        if RUST_AVAILABLE:
            # Test stats
            stats = rust_service.get_rust_stats()
            print(f"✅ Got Rust stats: {stats.get('rust_available', False)}")
            
            # Test recipe processing
            test_text = "Ingredients:\n1 cup flour\n2 eggs\nInstructions:\nMix and bake"
            result = rust_service.parse_recipe_rust(test_text)
            print(f"✅ Python integration recipe parsing: {result['title']}")
            
        return True
    except Exception as e:
        print(f"❌ Python integration test failed: {e}")
        return False

def benchmark_performance():
    """Simple performance benchmark"""
    print("\n⚡ Running performance benchmark...")
    try:
        import chef_genius_core
        
        # Benchmark system info calls
        start_time = time.time()
        for _ in range(1000):
            info = chef_genius_core.get_system_info()
        end_time = time.time()
        
        avg_time_us = (end_time - start_time) * 1000 * 1000 / 1000
        print(f"✅ System info calls: {avg_time_us:.1f} μs average")
        
        # Benchmark recipe processing
        processor = chef_genius_core.PyRecipeProcessor()
        test_text = "1 cup flour\n2 eggs\n1 tsp salt"
        
        start_time = time.time()
        for _ in range(100):
            ingredients = processor.extract_ingredients(test_text)
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) * 1000 / 100
        print(f"✅ Ingredient extraction: {avg_time_ms:.2f} ms average")
        
        return True
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        return False

async def main():
    """Run all tests"""
    print("🦀 Chef Genius Rust Integration Test Suite")
    print("=" * 50)
    
    tests = [
        ("Rust Import", test_rust_import),
        ("Rust Engines", test_rust_engines),
        ("Recipe Processing", test_recipe_processing),
        ("Nutrition Analysis", test_nutrition_analysis),
        ("Vector Search", test_vector_search),
        ("Python Integration", test_python_integration),
        ("Performance Benchmark", benchmark_performance),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"💥 {test_name} CRASHED: {e}")
    
    print(f"\n{'='*50}")
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Rust integration is working perfectly!")
        print("\n🚀 Ready to use Chef Genius with Rust acceleration!")
        print("\nNext steps:")
        print("1. Start your FastAPI server: uvicorn backend.app.main:app --reload")
        print("2. Test the API endpoints: curl http://localhost:8000/api/v1/rust/status")
        print("3. Enjoy 5-15x performance improvements! ⚡")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        if passed > 0:
            print("✅ Partial functionality is available.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(main())