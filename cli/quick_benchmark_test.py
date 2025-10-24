#!/usr/bin/env python3
"""
Quick benchmark test of your most promising datasets
Tests the Rust-powered data loader on your actual data
"""

import time
from pathlib import Path

# Import the fast data loader
try:
    from fast_dataloader import FastDataLoader, DataLoaderConfig, benchmark_dataloader
    FAST_LOADER_AVAILABLE = True
    print("🦀 Rust data loader available!")
except ImportError:
    FAST_LOADER_AVAILABLE = False
    print("❌ Fast data loader not available")

def test_promising_datasets():
    """Test your most promising datasets"""
    
    # List of your most promising datasets based on the folder structure
    test_datasets = [
        "data/datasets/allrecipes_250k/training.json",
        "data/datasets/recipe_nlg/training.json", 
        "data/datasets/recipe_dataset_simple/training.json",
        "data/datasets/indian_food_analysis/training.json",
        "data/datasets/recipe_box/training.json",
        "data/datasets/indian_recipe_api/training.json",
        "data/datasets/sample_recipes/training.json",
        
        # Large CSV datasets
        "data/datasets/food_com_recipes_2m/recipes_data.csv",
        "data/datasets/recipe_nlg/RecipeNLG_dataset.csv",
        "data/datasets/indian_food_6k/IndianFoodDatasetCSV.csv",
        "data/datasets/food_recipes_8k/food_recipes.csv",
    ]
    
    print("🚀 TESTING YOUR BEST DATASETS WITH RUST-POWERED LOADER")
    print("=" * 65)
    
    results = []
    
    for dataset_path in test_datasets:
        path = Path(dataset_path)
        
        if not path.exists():
            print(f"⏭️  Skipping {path.name} (not found)")
            continue
        
        print(f"\n🔥 Testing: {path.parent.name}/{path.name}")
        
        # Check file size
        try:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"   Size: {size_mb:.1f}MB")
        except:
            size_mb = 0
        
        if not FAST_LOADER_AVAILABLE:
            print("   ❌ Fast loader not available")
            continue
        
        # Test with fast loader
        try:
            config = DataLoaderConfig(
                batch_size=8,
                shuffle=True,
                buffer_size=32,
                num_prefetch_threads=6,
                use_rust=True
            )
            
            start_time = time.time()
            loader = FastDataLoader(str(path), config)
            init_time = time.time() - start_time
            
            # Quick performance test
            start_time = time.time()
            batch_count = 0
            total_samples = 0
            
            for batch in loader:
                batch_count += 1
                total_samples += len(batch.get('input_ids', []))
                
                if batch_count >= 25:  # Test 25 batches
                    break
            
            test_time = time.time() - start_time
            
            if test_time > 0:
                samples_per_sec = total_samples / test_time
                avg_batch_time = test_time / batch_count * 1000 if batch_count > 0 else 0
                
                print(f"   ✅ {samples_per_sec:.1f} samples/sec, {avg_batch_time:.1f}ms/batch")
                
                # Performance rating
                if samples_per_sec >= 30:
                    rating = "🔥 EXCELLENT"
                elif samples_per_sec >= 15:
                    rating = "✅ VERY GOOD"
                elif samples_per_sec >= 8:
                    rating = "✅ GOOD"
                elif samples_per_sec >= 3:
                    rating = "⚠️  FAIR"
                else:
                    rating = "❌ SLOW"
                
                print(f"   {rating}")
                
                results.append({
                    'path': str(path),
                    'name': f"{path.parent.name}/{path.name}",
                    'size_mb': size_mb,
                    'samples_per_sec': samples_per_sec,
                    'avg_batch_time': avg_batch_time,
                    'total_samples': total_samples,
                    'rating': rating
                })
            else:
                print("   ❌ No batches loaded")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Show summary
    if results:
        print("\n" + "=" * 65)
        print("🏆 PERFORMANCE SUMMARY (Best to Worst)")
        print("=" * 65)
        
        # Sort by performance
        results.sort(key=lambda x: x['samples_per_sec'], reverse=True)
        
        for i, result in enumerate(results[:10], 1):  # Top 10
            print(f"{i:2d}. {result['name']}")
            print(f"    🚀 {result['samples_per_sec']:.1f} samples/sec ({result['avg_batch_time']:.1f}ms/batch)")
            print(f"    📊 {result['rating']}")
            print()
        
        # Best dataset recommendation
        best = results[0]
        print("🎯 RECOMMENDED FOR TRAINING:")
        print(f"   Use: {best['path']}")
        print(f"   Performance: {best['samples_per_sec']:.1f} samples/sec")
        print()
        
        # Show integration code
        print("🔧 INTEGRATION CODE:")
        print("```python")
        print("from dataloader_integration import create_optimized_dataloader")
        print()
        print("# Replace your slow DataLoader with:")
        print("fast_loader = create_optimized_dataloader(")
        print(f'    data_path="{best["path"]}",')
        print("    tokenizer=tokenizer,")
        print("    batch_size=8,")
        print("    use_rust=True")
        print(")")
        print("```")
        
        # Expected improvement
        original_speed = 0.01  # Your current speed
        improvement = best['samples_per_sec'] / original_speed
        print(f"\n🎊 IMPROVEMENT: {improvement:.0f}x faster than your current 0.01 samples/sec!")
        print("Your Discord notifications will show these improved metrics! 🎉")
        
    else:
        print("\n❌ No datasets could be tested successfully")
        print("Make sure you have training data files in the correct locations")

def main():
    print("🔥 QUICK PERFORMANCE TEST OF YOUR DATASETS")
    print("Testing Rust-powered data loader on your actual data...")
    print()
    
    test_promising_datasets()

if __name__ == "__main__":
    main()