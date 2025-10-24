#!/usr/bin/env python3
"""
🦀 RUST-POWERED DATASET TRANSFORMER
Transform 2.2M recipes to B2B format using Rust parallel processing
Optimized for Ryzen 9 3900X with quality validation
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional

# Try to import Rust core
try:
    import chef_genius_core
    RUST_AVAILABLE = True
    print("🦀 Rust dataset transformer loaded successfully")
except ImportError as e:
    RUST_AVAILABLE = False
    print(f"❌ Rust core not available: {e}")
    print("📝 Build with: cd chef_genius_core && cargo build --release --no-default-features")
    exit(1)

class RustDatasetTransformer:
    """High-performance dataset transformer using Rust backend."""
    
    def __init__(self):
        if not RUST_AVAILABLE:
            raise ImportError("Rust core is required for dataset transformation")
        
        self.transformer = chef_genius_core.PyDatasetTransformer()
        print("🦀 Rust transformer initialized")
    
    def transform_large_dataset(
        self,
        dataset_path: str,
        output_path: str,
        b2b_percentage: float = 0.2,
        max_recipes: Optional[int] = None,
        benchmark: bool = False
    ) -> Dict:
        """Transform large dataset with Rust performance."""
        
        print(f"🏭 RUST DATASET TRANSFORMATION")
        print(f"=" * 50)
        print(f"📁 Input: {dataset_path}")
        print(f"📁 Output: {output_path}")
        print(f"🏢 B2B percentage: {b2b_percentage*100}%")
        
        if max_recipes:
            print(f"🎯 Max recipes: {max_recipes:,}")
        
        # Load dataset
        print(f"\n📊 Loading dataset...")
        start_time = time.time()
        
        with open(dataset_path, 'r') as f:
            dataset_json = f.read()
        
        load_time = time.time() - start_time
        
        # Get dataset size
        dataset_preview = json.loads(dataset_json)
        total_recipes = len(dataset_preview)
        
        print(f"✅ Loaded {total_recipes:,} recipes in {load_time:.2f}s")
        
        if max_recipes and max_recipes < total_recipes:
            print(f"🎯 Sampling {max_recipes:,} recipes for processing")
        
        # Run benchmark if requested
        if benchmark:
            print(f"\n🏎️  Running performance benchmark...")
            benchmark_results = self.transformer.benchmark_performance(1000)
            
            print(f"📊 BENCHMARK RESULTS:")
            for key, speed in benchmark_results.items():
                print(f"   {key}: {speed:.1f} recipes/sec")
        
        # Transform dataset
        print(f"\n🦀 Starting Rust transformation...")
        start_time = time.time()
        
        try:
            transformed_json = self.transformer.transform_dataset_parallel(
                dataset_json,
                b2b_percentage,
                max_recipes
            )
            
            transformation_time = time.time() - start_time
            
            # Parse results for analysis
            transformed_data = json.loads(transformed_json)
            
            print(f"\n⚡ TRANSFORMATION COMPLETE")
            print(f"Time: {transformation_time:.2f}s")
            print(f"Speed: {len(transformed_data)/transformation_time:.1f} recipes/sec")
            print(f"Processed: {len(transformed_data):,} recipes")
            
            # Save results
            print(f"\n💾 Saving results...")
            save_start = time.time()
            
            with open(output_path, 'w') as f:
                f.write(transformed_json)
            
            save_time = time.time() - save_start
            print(f"✅ Saved to {output_path} in {save_time:.2f}s")
            
            # Generate quality report
            quality_report = self._generate_quality_report(transformed_data)
            quality_path = output_path.replace('.json', '_quality_report.json')
            
            with open(quality_path, 'w') as f:
                json.dump(quality_report, f, indent=2)
            
            print(f"📊 Quality report: {quality_path}")
            
            return {
                "total_recipes": len(transformed_data),
                "transformation_time": transformation_time,
                "speed": len(transformed_data) / transformation_time,
                "output_file": output_path,
                "quality_report": quality_report
            }
            
        except Exception as e:
            print(f"❌ Transformation failed: {e}")
            raise
    
    def _generate_quality_report(self, transformed_data: List[Dict]) -> Dict:
        """Generate comprehensive quality report."""
        
        b2b_recipes = [r for r in transformed_data if r.get('format') == 'b2b_enterprise']
        regular_recipes = [r for r in transformed_data if r.get('format') == 'simple']
        
        # Quality metrics analysis
        b2b_quality_scores = [
            r['quality_metrics']['complexity_score'] 
            for r in b2b_recipes 
            if 'quality_metrics' in r
        ]
        
        regular_quality_scores = [
            r['quality_metrics']['complexity_score']
            for r in regular_recipes
            if 'quality_metrics' in r
        ]
        
        b2b_features = [
            r['quality_metrics']['b2b_features_count']
            for r in b2b_recipes
            if 'quality_metrics' in r
        ]
        
        # Business scenario distribution
        scenario_distribution = {}
        for recipe in b2b_recipes:
            scenario = recipe.get('business_scenario', 'unknown')
            scenario_distribution[scenario] = scenario_distribution.get(scenario, 0) + 1
        
        # Volume distribution
        volumes = [r.get('volume', 0) for r in b2b_recipes if r.get('volume')]
        
        report = {
            "summary": {
                "total_recipes": len(transformed_data),
                "b2b_count": len(b2b_recipes),
                "regular_count": len(regular_recipes),
                "b2b_percentage": len(b2b_recipes) / len(transformed_data) * 100 if transformed_data else 0
            },
            "quality_metrics": {
                "avg_b2b_quality": sum(b2b_quality_scores) / len(b2b_quality_scores) if b2b_quality_scores else 0,
                "avg_regular_quality": sum(regular_quality_scores) / len(regular_quality_scores) if regular_quality_scores else 0,
                "avg_b2b_features": sum(b2b_features) / len(b2b_features) if b2b_features else 0,
                "high_quality_b2b": sum(1 for score in b2b_quality_scores if score >= 50),
                "high_quality_regular": sum(1 for score in regular_quality_scores if score >= 50)
            },
            "business_distribution": scenario_distribution,
            "volume_stats": {
                "min_volume": min(volumes) if volumes else 0,
                "max_volume": max(volumes) if volumes else 0,
                "avg_volume": sum(volumes) / len(volumes) if volumes else 0
            }
        }
        
        return report
    
    def quick_test(self, sample_size: int = 1000) -> Dict:
        """Quick test with sample data."""
        print(f"🧪 QUICK TEST - {sample_size} recipes")
        
        # Create test dataset
        test_dataset = []
        for i in range(sample_size):
            test_dataset.append({
                "title": f"Test Recipe {i+1}",
                "ingredients": [
                    "2 cups flour",
                    "1 cup sugar",
                    "3 eggs",
                    "1/2 cup butter",
                    "1 tsp vanilla extract"
                ],
                "instructions": [
                    "Preheat oven to 350°F",
                    "Mix dry ingredients in large bowl", 
                    "Add wet ingredients and mix well",
                    "Bake for 25-30 minutes until golden"
                ],
                "cuisine": "American" if i % 2 == 0 else "Italian"
            })
        
        test_json = json.dumps(test_dataset)
        
        # Transform with different B2B percentages
        results = {}
        
        for b2b_pct in [0.1, 0.2, 0.5]:
            print(f"\n🦀 Testing {b2b_pct*100}% B2B conversion...")
            
            start_time = time.time()
            transformed_json = self.transformer.transform_dataset_parallel(
                test_json, b2b_pct, None
            )
            duration = time.time() - start_time
            
            transformed_data = json.loads(transformed_json)
            speed = len(transformed_data) / duration
            
            results[f"{int(b2b_pct*100)}pct_b2b"] = {
                "recipes": len(transformed_data),
                "time": duration,
                "speed": speed,
                "b2b_count": len([r for r in transformed_data if r['format'] == 'b2b_enterprise'])
            }
            
            print(f"   ⚡ {speed:.1f} recipes/sec, {duration:.3f}s total")
        
        return results

def main():
    """Main transformation workflow."""
    print("🦀 RUST DATASET TRANSFORMER FOR 2.2M RECIPES")
    print("=" * 60)
    
    if not RUST_AVAILABLE:
        print("❌ Rust core required. Build with:")
        print("   cd chef_genius_core && cargo build --release --no-default-features")
        return
    
    # Initialize transformer
    transformer = RustDatasetTransformer()
    
    # Dataset options
    dataset_files = [
        "/Users/timmy/workspace/ai-apps/chef-genius/cli/allrecipes_250k/training.json",
        "/Users/timmy/workspace/ai-apps/chef-genius/data/recipes.json"
    ]
    
    print("\n📁 Available datasets:")
    for i, file_path in enumerate(dataset_files, 1):
        path = Path(file_path)
        if path.exists():
            size_mb = path.stat().st_size / (1024 * 1024)
            print(f"{i}. {path.name} ({size_mb:.1f} MB)")
        else:
            print(f"{i}. {path.name} (NOT FOUND)")
    
    print(f"\n🎯 Transformation Options:")
    print(f"1. 🧪 Quick Test (1,000 recipes)")
    print(f"2. 📊 Small Scale (10,000 recipes, 20% B2B)")
    print(f"3. 🏢 Medium Scale (100,000 recipes, 20% B2B)")
    print(f"4. 🚀 Full Scale (All recipes, 20% B2B)")
    print(f"5. 🏎️  Benchmark Only")
    
    choice = input("\nChoose option (1-5): ").strip()
    
    if choice == "1":
        # Quick test
        results = transformer.quick_test(1000)
        print(f"\n🎉 QUICK TEST RESULTS:")
        for test_name, metrics in results.items():
            print(f"   {test_name}: {metrics['speed']:.1f} recipes/sec")
    
    elif choice == "2":
        # Small scale
        output_file = "b2b_dataset_small_10k.json"
        results = transformer.transform_large_dataset(
            dataset_files[0], output_file, 
            b2b_percentage=0.2, max_recipes=10000
        )
        print(f"\n🎉 Small scale complete: {results['speed']:.1f} recipes/sec")
    
    elif choice == "3":
        # Medium scale  
        output_file = "b2b_dataset_medium_100k.json"
        results = transformer.transform_large_dataset(
            dataset_files[0], output_file,
            b2b_percentage=0.2, max_recipes=100000
        )
        print(f"\n🎉 Medium scale complete: {results['speed']:.1f} recipes/sec")
    
    elif choice == "4":
        # Full scale
        output_file = "b2b_dataset_full_2M.json"
        results = transformer.transform_large_dataset(
            dataset_files[0], output_file,
            b2b_percentage=0.2, max_recipes=None
        )
        print(f"\n🎉 Full scale complete: {results['speed']:.1f} recipes/sec")
        print(f"🚀 Your Ryzen 9 3900X processed {results['total_recipes']:,} recipes!")
    
    elif choice == "5":
        # Benchmark only
        benchmark_results = transformer.transformer.benchmark_performance(5000)
        print(f"\n🏎️  RYZEN 9 3900X BENCHMARK RESULTS:")
        for test, speed in benchmark_results.items():
            print(f"   {test}: {speed:.1f} recipes/sec")
    
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    main()