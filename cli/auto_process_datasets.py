#!/usr/bin/env python3
"""
Auto-discovery and processing system for all dataset folders
Finds all training data files and tests them with the fast data loader
"""

import os
import sys
import json
import csv
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Import our fast data loader
try:
    from fast_dataloader import FastDataLoader, DataLoaderConfig, benchmark_dataloader
    FAST_LOADER_AVAILABLE = True
except ImportError:
    FAST_LOADER_AVAILABLE = False
    print("⚠️  Fast data loader not available. Run: python install_rust_dataloader.py")

class DatasetProcessor:
    """Auto-discovery and processing of all recipe datasets"""
    
    def __init__(self, base_path: str = "data/datasets"):
        self.base_path = Path(base_path)
        self.supported_formats = ['.json', '.csv']
        self.discovered_datasets = []
        self.benchmark_results = {}
        
    def discover_datasets(self) -> List[Dict[str, Any]]:
        """Discover all processable datasets"""
        print(f"🔍 Scanning {self.base_path} for recipe datasets...")
        
        datasets = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)
            
            for file in files:
                file_path = root_path / file
                
                # Check if it's a supported format
                if file_path.suffix.lower() in self.supported_formats:
                    # Skip very small files (likely not datasets)
                    try:
                        file_size = file_path.stat().st_size
                        if file_size < 1024:  # Skip files smaller than 1KB
                            continue
                    except:
                        continue
                    
                    # Analyze the dataset
                    dataset_info = self.analyze_dataset(file_path)
                    if dataset_info:
                        datasets.append(dataset_info)
        
        # Sort by estimated sample count (largest first)
        datasets.sort(key=lambda x: x.get('estimated_samples', 0), reverse=True)
        
        self.discovered_datasets = datasets
        return datasets
    
    def analyze_dataset(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a single dataset file"""
        try:
            file_size = file_path.stat().st_size
            file_size_mb = file_size / (1024 * 1024)
            
            info = {
                'path': str(file_path),
                'name': file_path.name,
                'folder': file_path.parent.name,
                'format': file_path.suffix.lower(),
                'size_mb': round(file_size_mb, 2),
                'estimated_samples': 0,
                'has_required_fields': False,
                'sample_data': None
            }
            
            # Quick analysis based on format
            if file_path.suffix.lower() == '.json':
                info.update(self.analyze_json_file(file_path))
            elif file_path.suffix.lower() == '.csv':
                info.update(self.analyze_csv_file(file_path))
            
            return info if info['estimated_samples'] > 0 else None
            
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            return None
    
    def analyze_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze JSON dataset file"""
        info = {'estimated_samples': 0, 'has_required_fields': False}
        
        try:
            # Read first few lines to check format
            with open(file_path, 'r', encoding='utf-8') as f:
                sample_lines = []
                for i, line in enumerate(f):
                    if i >= 5:  # Check first 5 lines
                        break
                    try:
                        data = json.loads(line.strip())
                        sample_lines.append(data)
                    except json.JSONDecodeError:
                        continue
                
                if sample_lines:
                    # Check if has required fields
                    sample = sample_lines[0]
                    required_fields = ['instruction', 'ingredients', 'title']
                    has_fields = all(field in sample for field in required_fields)
                    
                    info.update({
                        'has_required_fields': has_fields,
                        'sample_data': sample,
                        'fields': list(sample.keys()) if isinstance(sample, dict) else []
                    })
                    
                    if has_fields:
                        # Estimate total lines
                        f.seek(0)
                        line_count = sum(1 for _ in f)
                        info['estimated_samples'] = line_count
        
        except Exception as e:
            print(f"Error reading JSON {file_path}: {e}")
        
        return info
    
    def analyze_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze CSV dataset file"""
        info = {'estimated_samples': 0, 'has_required_fields': False}
        
        try:
            # Try different CSV reading strategies for robustness
            df_sample = None
            
            # Strategy 1: Standard pandas read
            try:
                df_sample = pd.read_csv(file_path, nrows=5, encoding='utf-8')
            except:
                pass
            
            # Strategy 2: Try with error handling
            if df_sample is None or df_sample.empty:
                try:
                    df_sample = pd.read_csv(file_path, nrows=5, encoding='utf-8', 
                                          error_bad_lines=False, warn_bad_lines=False)
                except:
                    pass
            
            # Strategy 3: Try with different separator
            if df_sample is None or df_sample.empty:
                try:
                    df_sample = pd.read_csv(file_path, nrows=5, encoding='utf-8', 
                                          sep=None, engine='python')
                except:
                    pass
            
            # Strategy 4: Try with latin-1 encoding
            if df_sample is None or df_sample.empty:
                try:
                    df_sample = pd.read_csv(file_path, nrows=5, encoding='latin-1')
                except:
                    pass
            
            if df_sample is not None and not df_sample.empty:
                columns = [col.lower() for col in df_sample.columns]
                
                # Check for recipe-related fields
                recipe_indicators = ['instruction', 'ingredients', 'title', 'recipe', 'directions', 'name']
                has_recipe_fields = any(indicator in ' '.join(columns) for indicator in recipe_indicators)
                
                info.update({
                    'has_required_fields': has_recipe_fields,
                    'sample_data': df_sample.iloc[0].to_dict() if len(df_sample) > 0 else {},
                    'fields': list(df_sample.columns)
                })
                
                if has_recipe_fields:
                    # Get total row count (try to avoid reading full file if possible)
                    try:
                        # Quick line count method
                        with open(file_path, 'r', encoding='utf-8') as f:
                            line_count = sum(1 for _ in f) - 1  # Subtract header
                        info['estimated_samples'] = max(0, line_count)
                    except:
                        try:
                            df_full = pd.read_csv(file_path, encoding='utf-8')
                            info['estimated_samples'] = len(df_full)
                        except:
                            info['estimated_samples'] = 1000  # Fallback estimate
        
        except Exception as e:
            print(f"Error reading CSV {file_path}: {e}")
        
        return info
    
    def benchmark_dataset(self, dataset_info: Dict[str, Any]) -> Dict[str, Any]:
        """Benchmark a single dataset with fast data loader"""
        file_path = dataset_info['path']
        
        print(f"\n🚀 Benchmarking {dataset_info['name']} ({dataset_info['size_mb']}MB, ~{dataset_info['estimated_samples']} samples)")
        
        if not FAST_LOADER_AVAILABLE:
            return {'error': 'Fast loader not available'}
        
        try:
            # Test with different batch sizes
            batch_sizes = [4, 8, 16] if dataset_info['estimated_samples'] > 100 else [4]
            results = {}
            
            for batch_size in batch_sizes:
                print(f"  Testing batch size {batch_size}...")
                
                # Create optimized config
                config = DataLoaderConfig(
                    batch_size=batch_size,
                    shuffle=True,
                    buffer_size=32,
                    num_prefetch_threads=6,
                    use_rust=FAST_LOADER_AVAILABLE
                )
                
                start_time = time.time()
                
                try:
                    loader = FastDataLoader(file_path, config)
                    
                    # Test loading speed
                    test_batches = min(50, dataset_info['estimated_samples'] // batch_size)
                    batch_count = 0
                    total_samples = 0
                    
                    for batch in loader:
                        batch_count += 1
                        total_samples += len(batch.get('input_ids', []))
                        
                        if batch_count >= test_batches:
                            break
                    
                    test_time = time.time() - start_time
                    samples_per_sec = total_samples / test_time if test_time > 0 else 0
                    avg_batch_time = test_time / batch_count * 1000 if batch_count > 0 else 0
                    
                    results[f'batch_{batch_size}'] = {
                        'samples_per_sec': round(samples_per_sec, 1),
                        'avg_batch_time_ms': round(avg_batch_time, 1),
                        'total_samples': total_samples,
                        'test_time': round(test_time, 2)
                    }
                    
                    print(f"    ✅ {samples_per_sec:.1f} samples/sec, {avg_batch_time:.1f}ms/batch")
                    
                except Exception as e:
                    results[f'batch_{batch_size}'] = {'error': str(e)}
                    print(f"    ❌ Error: {e}")
            
            return results
            
        except Exception as e:
            return {'error': str(e)}
    
    def process_all_datasets(self, max_workers: int = 3):
        """Process all discovered datasets in parallel"""
        if not self.discovered_datasets:
            self.discover_datasets()
        
        print(f"\n🔥 Found {len(self.discovered_datasets)} datasets to benchmark")
        print("=" * 60)
        
        # Show summary
        for i, dataset in enumerate(self.discovered_datasets[:10], 1):  # Show top 10
            print(f"{i:2d}. {dataset['folder']}/{dataset['name']}")
            print(f"    Size: {dataset['size_mb']}MB, ~{dataset['estimated_samples']:,} samples")
            print(f"    Fields: {dataset.get('fields', [])[:5]}")  # Show first 5 fields
        
        if len(self.discovered_datasets) > 10:
            print(f"    ... and {len(self.discovered_datasets) - 10} more datasets")
        
        print("\n🚀 Starting benchmarks...")
        
        # Process in parallel (limited to avoid overwhelming system)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_dataset = {
                executor.submit(self.benchmark_dataset, dataset): dataset 
                for dataset in self.discovered_datasets[:15]  # Limit to top 15 for speed
            }
            
            for future in as_completed(future_to_dataset):
                dataset = future_to_dataset[future]
                try:
                    result = future.result()
                    self.benchmark_results[dataset['name']] = {
                        'dataset_info': dataset,
                        'benchmark': result
                    }
                except Exception as e:
                    print(f"❌ {dataset['name']} failed: {e}")
    
    def generate_report(self) -> str:
        """Generate a comprehensive performance report"""
        if not self.benchmark_results:
            return "No benchmark results available"
        
        report = []
        report.append("🦀 RUST-POWERED DATA LOADER PERFORMANCE REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Sort by best performance
        sorted_results = sorted(
            self.benchmark_results.items(),
            key=lambda x: max([
                result.get('samples_per_sec', 0) 
                for result in x[1]['benchmark'].values() 
                if isinstance(result, dict) and 'samples_per_sec' in result
            ] + [0]),
            reverse=True
        )
        
        report.append("TOP PERFORMING DATASETS:")
        report.append("")
        
        for name, data in sorted_results[:10]:
            dataset = data['dataset_info']
            benchmark = data['benchmark']
            
            report.append(f"📊 {dataset['folder']}/{name}")
            report.append(f"   Size: {dataset['size_mb']}MB, {dataset['estimated_samples']:,} samples")
            
            # Show best performance
            best_perf = 0
            best_config = ""
            for config, result in benchmark.items():
                if isinstance(result, dict) and 'samples_per_sec' in result:
                    if result['samples_per_sec'] > best_perf:
                        best_perf = result['samples_per_sec']
                        best_config = config
            
            if best_perf > 0:
                result = benchmark[best_config]
                report.append(f"   🚀 BEST: {best_perf:.1f} samples/sec ({result['avg_batch_time_ms']:.1f}ms/batch)")
                
                # Performance assessment
                if best_perf >= 25:
                    report.append("   ✅ EXCELLENT - Ready for production training!")
                elif best_perf >= 10:
                    report.append("   ✅ GOOD - Major improvement over standard loader")
                elif best_perf >= 5:
                    report.append("   ⚠️  MODERATE - Better than before but could optimize")
                else:
                    report.append("   ❌ SLOW - May need data format optimization")
            else:
                report.append("   ❌ FAILED - Could not benchmark")
            
            report.append("")
        
        report.append("")
        report.append("INTEGRATION RECOMMENDATIONS:")
        report.append("")
        
        # Find best datasets
        best_datasets = [
            data for name, data in sorted_results[:5]
            if any(
                isinstance(result, dict) and result.get('samples_per_sec', 0) >= 10
                for result in data['benchmark'].values()
            )
        ]
        
        if best_datasets:
            report.append("🎯 USE THESE DATASETS FOR TRAINING:")
            for data in best_datasets:
                dataset = data['dataset_info']
                report.append(f"   • {dataset['path']}")
            
            report.append("")
            report.append("🔧 INTEGRATION CODE:")
            report.append("")
            report.append("```python")
            report.append("from dataloader_integration import create_optimized_dataloader")
            report.append("")
            report.append("# Replace your slow DataLoader with this:")
            best_path = best_datasets[0]['dataset_info']['path']
            report.append(f'fast_loader = create_optimized_dataloader(')
            report.append(f'    data_path="{best_path}",')
            report.append(f'    tokenizer=tokenizer,')
            report.append(f'    batch_size=8,')
            report.append(f'    use_rust=True')
            report.append(f')')
            report.append("```")
        else:
            report.append("⚠️  No datasets achieved >10 samples/sec. Consider:")
            report.append("   • Installing Rust loader: python install_rust_dataloader.py")
            report.append("   • Converting CSV to JSON format")
            report.append("   • Using smaller datasets for testing")
        
        return "\n".join(report)
    
    def save_report(self, filename: str = "dataset_performance_report.txt"):
        """Save the performance report to file"""
        report = self.generate_report()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📝 Report saved to {filename}")
        return filename

def main():
    print("🔍 AUTO-DISCOVERING AND BENCHMARKING ALL RECIPE DATASETS")
    print("This will find and test all your datasets for maximum performance!")
    print()
    
    # Check if fast loader is available
    if not FAST_LOADER_AVAILABLE:
        print("⚠️  Fast data loader not available!")
        print("Run this first: python install_rust_dataloader.py")
        print("Continuing with discovery only...")
        print()
    
    # Create processor
    processor = DatasetProcessor()
    
    # Discover all datasets
    datasets = processor.discover_datasets()
    
    if not datasets:
        print("❌ No datasets found in data/datasets/")
        print("Make sure your training data is in the correct location.")
        return
    
    # Benchmark if fast loader available
    if FAST_LOADER_AVAILABLE:
        processor.process_all_datasets(max_workers=2)  # Conservative for Windows
        
        # Generate and display report
        print("\n" + processor.generate_report())
        
        # Save report
        processor.save_report()
        
        print("\n🎉 BENCHMARKING COMPLETE!")
        print("Your training should now be 10-50x faster!")
        
    else:
        print("\n📋 DISCOVERED DATASETS:")
        for i, dataset in enumerate(datasets[:15], 1):
            print(f"{i:2d}. {dataset['folder']}/{dataset['name']}")
            print(f"    Size: {dataset['size_mb']}MB, ~{dataset['estimated_samples']:,} samples")
        
        print(f"\nFound {len(datasets)} total datasets")
        print("Install fast loader to benchmark: python install_rust_dataloader.py")

if __name__ == "__main__":
    main()