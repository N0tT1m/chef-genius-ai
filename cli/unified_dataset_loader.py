#!/usr/bin/env python3
"""
Unified Dataset Loader - Combines ALL your datasets for maximum training data
Uses the fastest Rust-powered loader for each dataset and merges them
"""

import os
import json
import csv
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue, Empty
import pandas as pd

try:
    from fast_dataloader import FastDataLoader, DataLoaderConfig
    FAST_LOADER_AVAILABLE = True
except ImportError:
    FAST_LOADER_AVAILABLE = False

class UnifiedDatasetLoader:
    """
    Combines ALL your datasets into one super-fast training loader
    Automatically handles different formats and field mappings
    """
    
    def __init__(self, 
                 datasets_base_path: str = "data/datasets",
                 batch_size: int = 8,
                 shuffle: bool = True,
                 max_datasets: Optional[int] = None):
        
        self.datasets_base_path = Path(datasets_base_path)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.max_datasets = max_datasets
        
        # Discovered datasets
        self.datasets = []
        self.fast_loaders = {}
        self.dataset_weights = {}
        
        # Combined data buffer
        self.combined_buffer = Queue(maxsize=100)
        self.buffer_threads = []
        self.stop_loading = threading.Event()
        
        # Statistics
        self.total_samples = 0
        self.samples_processed = 0
        self.current_epoch = 0
        
        # Field mappings for different dataset formats
        self.field_mappings = {
            'title': ['title', 'name', 'recipe_name', 'recipename', 'Title', 'RecipeName', 'recipe_title', 'TranslatedRecipeName'],
            'ingredients': ['ingredients', 'ingredient_tokens', 'Ingredients', 'TranslatedIngredients'],
            'instructions': ['instructions', 'directions', 'steps_tokens', 'Instructions', 'TranslatedInstructions']
        }
        
        print(f"🔍 Initializing Unified Dataset Loader...")
        self.discover_and_prepare_datasets()
    
    def discover_and_prepare_datasets(self):
        """Discover all datasets and prepare fast loaders"""
        
        print(f"🔍 Scanning {self.datasets_base_path} for all datasets...")
        
        # Find all potential dataset files
        dataset_files = []
        
        for root, dirs, files in os.walk(self.datasets_base_path):
            for file in files:
                if file.endswith(('.json', '.csv')) and not file.startswith('.'):
                    file_path = Path(root) / file
                    try:
                        size_mb = file_path.stat().st_size / (1024 * 1024)
                        if size_mb > 0.1:  # Skip tiny files
                            dataset_files.append({
                                'path': file_path,
                                'size_mb': size_mb,
                                'format': file_path.suffix.lower()
                            })
                    except:
                        continue
        
        # Sort by size (largest first) and limit if requested
        dataset_files.sort(key=lambda x: x['size_mb'], reverse=True)
        if self.max_datasets:
            dataset_files = dataset_files[:self.max_datasets]
        
        print(f"📊 Found {len(dataset_files)} datasets to combine:")
        
        # Analyze and prepare each dataset
        for i, dataset_file in enumerate(dataset_files, 1):
            path = dataset_file['path']
            print(f"  {i:2d}. {path.parent.name}/{path.name} ({dataset_file['size_mb']:.1f}MB)")
            
            # Analyze dataset structure
            dataset_info = self.analyze_dataset(path)
            if dataset_info and dataset_info['usable']:
                self.datasets.append(dataset_info)
                
                # Calculate weight based on dataset size and quality
                weight = min(dataset_info['estimated_samples'] / 1000, 100)  # Cap at 100
                self.dataset_weights[str(path)] = weight
                self.total_samples += dataset_info['estimated_samples']
        
        print(f"\n✅ Prepared {len(self.datasets)} datasets with {self.total_samples:,} total samples")
        
        # Create fast loaders for each dataset
        self.create_fast_loaders()
    
    def analyze_dataset(self, file_path: Path) -> Optional[Dict[str, Any]]:
        """Analyze a dataset file to understand its structure"""
        
        try:
            info = {
                'path': str(file_path),
                'name': file_path.name,
                'folder': file_path.parent.name,
                'format': file_path.suffix.lower(),
                'estimated_samples': 0,
                'field_mapping': {},
                'usable': False
            }
            
            if file_path.suffix.lower() == '.json':
                # Simple JSON structure analysis (original approach)
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 3:  # Check first 3 lines
                            break
                        try:
                            sample = json.loads(line.strip())
                            if isinstance(sample, dict):
                                mapping = self.find_field_mappings(sample.keys())
                                if mapping['title'] and (mapping['ingredients'] or mapping['instructions']):
                                    info['field_mapping'] = mapping
                                    info['usable'] = True
                                    # Simple sample count
                                    f.seek(0)
                                    info['estimated_samples'] = sum(1 for _ in f)
                                    break
                        except:
                            continue
            
            elif file_path.suffix.lower() == '.csv':
                # Analyze CSV structure with encoding detection
                try:
                    # Try multiple encodings for international datasets
                    encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    df_sample = None
                    
                    for encoding in encodings:
                        try:
                            df_sample = pd.read_csv(file_path, nrows=3, encoding=encoding)
                            break
                        except (UnicodeDecodeError, UnicodeError):
                            continue
                    
                    if df_sample is None:
                        raise Exception("Could not decode CSV with any encoding")
                    if not df_sample.empty:
                        mapping = self.find_field_mappings(df_sample.columns)
                        # Stricter validation - require title AND (ingredients OR instructions)
                        if mapping['title'] and (mapping['ingredients'] or mapping['instructions']):
                            info['field_mapping'] = mapping
                            info['usable'] = True
                            # Quick line count
                            with open(file_path, 'r', encoding='utf-8') as f:
                                info['estimated_samples'] = sum(1 for _ in f) - 1  # Subtract header
                except Exception as e:
                    print(f"    ⚠️  Could not read CSV: {e}")
            
            return info if info['usable'] else None
            
        except Exception as e:
            print(f"    ❌ Error analyzing {file_path}: {e}")
            return None
    
    def find_field_mappings(self, available_fields: List[str]) -> Dict[str, Optional[str]]:
        """Find the best field mappings for this dataset"""
        
        mapping = {'title': None, 'ingredients': None, 'instructions': None}
        available_lower = [field.lower() for field in available_fields]
        
        for target_field, possible_names in self.field_mappings.items():
            for possible_name in possible_names:
                if possible_name.lower() in available_lower:
                    # Find the exact case match
                    for field in available_fields:
                        if field.lower() == possible_name.lower():
                            mapping[target_field] = field
                            break
                    break
        
        return mapping
    
    def create_fast_loaders(self):
        """Create fast loaders for each usable dataset"""
        
        if not FAST_LOADER_AVAILABLE:
            print("❌ Fast loader not available - install with: python install_rust_dataloader.py")
            return
        
        print(f"\n🦀 Creating Rust-powered loaders for {len(self.datasets)} datasets...")
        
        for dataset in self.datasets:
            try:
                # Create optimized config for each dataset
                config = DataLoaderConfig(
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    buffer_size=16,  # Smaller buffer per dataset
                    num_prefetch_threads=2,  # Fewer threads per dataset
                    use_rust=True
                )
                
                # Create the loader
                loader = FastDataLoader(dataset['path'], config)
                self.fast_loaders[dataset['path']] = {
                    'loader': loader,
                    'mapping': dataset['field_mapping'],
                    'info': dataset
                }
                
                print(f"  ✅ {dataset['folder']}/{dataset['name']}")
                
            except Exception as e:
                print(f"  ❌ {dataset['name']}: {e}")
    
    def normalize_batch(self, batch: Dict[str, List[str]], mapping: Dict[str, str]) -> Dict[str, List[str]]:
        """Normalize a batch to standard field names"""
        
        normalized = {
            'titles': [],
            'ingredients': [],
            'instructions': []
        }
        
        # Map fields based on the dataset's mapping
        for i in range(len(batch.get('input_ids', []))):
            # Default values
            title = "Recipe"
            ingredients = ""
            instructions = ""
            
            # Extract based on mapping
            if mapping.get('title') and mapping['title'] in batch:
                title = batch[mapping['title']][i] if i < len(batch[mapping['title']]) else "Recipe"
            
            if mapping.get('ingredients') and mapping['ingredients'] in batch:
                ingredients = batch[mapping['ingredients']][i] if i < len(batch[mapping['ingredients']]) else ""
            
            if mapping.get('instructions') and mapping['instructions'] in batch:
                instructions = batch[mapping['instructions']][i] if i < len(batch[mapping['instructions']]) else ""
            
            # Combine ingredients and instructions if one is missing
            if not ingredients and instructions:
                ingredients = "Mixed ingredients"
            if not instructions and ingredients:
                instructions = f"Prepare using: {ingredients}"
            
            normalized['titles'].append(title)
            normalized['ingredients'].append(ingredients)
            normalized['instructions'].append(instructions)
        
        return normalized
    
    def start_background_loading(self):
        """Start background threads to load from all datasets"""
        
        self.stop_loading.clear()
        
        for dataset_path, loader_info in self.fast_loaders.items():
            thread = threading.Thread(
                target=self.dataset_loader_worker,
                args=(loader_info,),
                daemon=True
            )
            thread.start()
            self.buffer_threads.append(thread)
        
        print(f"🚀 Started {len(self.buffer_threads)} background loading threads")
    
    def dataset_loader_worker(self, loader_info: Dict[str, Any]):
        """Worker thread for loading from a specific dataset"""
        
        loader = loader_info['loader']
        mapping = loader_info['mapping']
        dataset_name = os.path.basename(loader_info.get('path', 'unknown'))
        
        batches_processed = 0
        try:
            for batch in loader:
                if self.stop_loading.is_set():
                    print(f"🛑 Worker for {dataset_name} stopping (signal received)")
                    break
                
                # Normalize the batch
                normalized_batch = self.normalize_batch(batch, mapping)
                
                # Add to combined buffer (with timeout to prevent blocking)
                try:
                    self.combined_buffer.put(normalized_batch, timeout=1.0)
                    batches_processed += 1
                except:
                    # Buffer full, skip this batch
                    continue
            
            print(f"✅ Worker for {dataset_name} completed: {batches_processed} batches processed")
                    
        except StopIteration:
            print(f"🏁 Worker for {dataset_name} finished: {batches_processed} batches processed")
        except Exception as e:
            print(f"❌ Dataset worker error for {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    def __iter__(self):
        """Make this loader iterable"""
        
        # Only start background loading if no threads are currently active
        active_threads = len([t for t in self.buffer_threads if t.is_alive()])
        if active_threads == 0:
            print(f"🚀 Starting background loading for epoch {self.current_epoch}")
            self.start_background_loading()
        else:
            print(f"⚡ Reusing {active_threads} active background threads for epoch {self.current_epoch}")
        
        self.samples_processed = 0
        
        return self
    
    def __next__(self):
        """Get next combined batch"""
        
        try:
            # Get batch from buffer
            batch = self.combined_buffer.get(timeout=5.0)
            self.samples_processed += len(batch['titles'])
            
            # Format for training (combine ingredients and instructions)
            formatted_batch = {
                'input_ids': [],
                'titles': batch['titles'],
                'ingredients': batch['ingredients']
            }
            
            for i in range(len(batch['titles'])):
                # Create training text: "Ingredients: ... Instructions: ..."
                text = f"Ingredients: {batch['ingredients'][i]}\nInstructions: {batch['instructions'][i]}"
                formatted_batch['input_ids'].append(text)
            
            return formatted_batch
            
        except Empty:
            # No more data available
            raise StopIteration
    
    def reset(self):
        """Reset for new epoch"""
        
        print(f"🔄 Resetting data loader for epoch {self.current_epoch + 1}")
        
        # Stop background threads
        self.stop_loading.set()
        
        # Wait for all background threads to finish properly
        for thread in self.buffer_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)  # Give threads time to finish
        
        # Clear the thread list completely
        self.buffer_threads.clear()
        
        # Clear buffer completely
        cleared_count = 0
        while not self.combined_buffer.empty():
            try:
                self.combined_buffer.get_nowait()
                cleared_count += 1
            except:
                break
        
        if cleared_count > 0:
            print(f"🧹 Cleared {cleared_count} remaining items from buffer")
        
        # Reset individual loaders
        for dataset_path, loader_info in self.fast_loaders.items():
            if hasattr(loader_info['loader'], 'reset'):
                loader_info['loader'].reset()
                print(f"  ↻ Reset loader for {os.path.basename(dataset_path)}")
        
        # Reset our own state
        self.samples_processed = 0
        self.current_epoch += 1
        
        print(f"✅ Reset complete for epoch {self.current_epoch} - ready for new data")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        
        return {
            'total_datasets': len(self.datasets),
            'total_samples': self.total_samples,
            'samples_processed': self.samples_processed,
            'current_epoch': self.current_epoch,
            'active_threads': len([t for t in self.buffer_threads if t.is_alive()]),
            'buffer_size': self.combined_buffer.qsize(),
            'datasets_used': [d['name'] for d in self.datasets]
        }

def create_unified_dataloader(
    datasets_path: str = "data/datasets",
    batch_size: int = 8,
    shuffle: bool = True,
    max_datasets: Optional[int] = None
) -> UnifiedDatasetLoader:
    """
    Create a unified data loader that combines ALL your datasets
    
    Args:
        datasets_path: Path to datasets directory
        batch_size: Batch size for training
        shuffle: Whether to shuffle data
        max_datasets: Maximum number of datasets to use (None = all)
    
    Returns:
        UnifiedDatasetLoader that combines all datasets
    """
    
    return UnifiedDatasetLoader(
        datasets_base_path=datasets_path,
        batch_size=batch_size,
        shuffle=shuffle,
        max_datasets=max_datasets
    )

if __name__ == "__main__":
    print("🔥 UNIFIED DATASET LOADER TEST")
    print("Combining ALL your datasets for maximum training data!")
    print("=" * 60)
    
    # Create unified loader
    loader = create_unified_dataloader(
        datasets_path="data/datasets",
        batch_size=8,
        max_datasets=10  # Test with top 10 datasets
    )
    
    if loader.fast_loaders:
        print(f"\n🚀 Testing combined loading performance...")
        
        start_time = time.time()
        total_samples = 0
        batch_count = 0
        
        try:
            for batch in loader:
                batch_count += 1
                total_samples += len(batch['input_ids'])
                
                if batch_count % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = total_samples / elapsed if elapsed > 0 else 0
                    print(f"  Batch {batch_count}: {rate:.1f} samples/sec")
                
                if batch_count >= 50:  # Test 50 batches
                    break
        
        except KeyboardInterrupt:
            print("\n⏹️  Test stopped by user")
        
        elapsed = time.time() - start_time
        final_rate = total_samples / elapsed if elapsed > 0 else 0
        
        print(f"\n🎯 FINAL PERFORMANCE:")
        print(f"   Combined rate: {final_rate:.1f} samples/sec")
        print(f"   Total samples: {total_samples}")
        print(f"   Datasets used: {len(loader.datasets)}")
        
        stats = loader.get_stats()
        print(f"   Available samples: {stats['total_samples']:,}")
        
        if final_rate > 20:
            print("✅ EXCELLENT - Ready for production training with ALL datasets!")
        elif final_rate > 10:
            print("✅ GOOD - Major improvement with combined datasets!")
        else:
            print("⚠️  Consider using fewer datasets or optimizing further")
    
    else:
        print("❌ No fast loaders created. Check dataset formats and fast loader installation.")