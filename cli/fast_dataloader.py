import os
import sys
import time
from typing import Iterator, Dict, Any, Optional, List
import json
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from queue import Queue
import csv

try:
    from chef_genius_dataloader import FastDataLoader as RustFastDataLoader, create_fast_dataloader, benchmark_loading
    RUST_AVAILABLE = True
except ImportError:
    try:
        # Try alternative module name pattern
        import chef_genius_dataloader
        RustFastDataLoader = chef_genius_dataloader.FastDataLoader
        create_fast_dataloader = chef_genius_dataloader.create_fast_dataloader
        benchmark_loading = chef_genius_dataloader.benchmark_loading
        RUST_AVAILABLE = True
    except ImportError:
        RUST_AVAILABLE = False

@dataclass
class DataLoaderConfig:
    batch_size: int = 8
    shuffle: bool = True
    buffer_size: int = 32
    num_prefetch_threads: int = 4
    use_rust: bool = True

class PythonFastDataLoader:
    """Fallback Python implementation with optimizations"""
    
    def __init__(self, data_path: str, config: DataLoaderConfig):
        self.data_path = data_path
        self.config = config
        self.data: List[Dict[str, str]] = []
        self.indices: List[int] = []
        self.position = 0
        self.current_epoch = 0
        self.prefetch_queue = Queue(maxsize=config.buffer_size)
        self.prefetch_threads: List[threading.Thread] = []
        self.stop_prefetching = threading.Event()
        
        self._load_data()
        self._start_prefetching()
    
    def _load_data(self):
        """Load data with memory mapping for large files"""
        print(f"Loading data from {self.data_path}")
        start_time = time.time()
        
        if self.data_path.endswith('.json'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f):
                    try:
                        item = json.loads(line.strip())
                        if all(key in item for key in ['instruction', 'ingredients', 'title']):
                            self.data.append(item)
                    except json.JSONDecodeError as e:
                        if line_num < 10:  # Only log first few errors
                            print(f"JSON decode error on line {line_num}: {e}")
                        continue
        
        elif self.data_path.endswith('.csv'):
            with open(self.data_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if all(key in row for key in ['instruction', 'ingredients', 'title']):
                        self.data.append({
                            'instruction': row['instruction'],
                            'ingredients': row['ingredients'],
                            'title': row['title']
                        })
        
        self.indices = list(range(len(self.data)))
        if self.config.shuffle:
            np.random.shuffle(self.indices)
        
        load_time = time.time() - start_time
        print(f"Loaded {len(self.data)} samples in {load_time:.2f}s ({len(self.data)/load_time:.0f} samples/sec)")
    
    def _start_prefetching(self):
        """Start background threads for prefetching batches"""
        self.stop_prefetching.clear()
        
        for i in range(self.config.num_prefetch_threads):
            thread = threading.Thread(target=self._prefetch_worker, daemon=True)
            thread.start()
            self.prefetch_threads.append(thread)
    
    def _prefetch_worker(self):
        """Worker thread for prefetching batches"""
        while not self.stop_prefetching.is_set():
            try:
                batch = self._get_next_batch_sync()
                if batch is None:
                    break
                
                if not self.prefetch_queue.full():
                    self.prefetch_queue.put(batch, timeout=1.0)
                else:
                    time.sleep(0.001)  # Brief pause if queue is full
                    
            except Exception as e:
                print(f"Prefetch worker error: {e}")
                break
    
    def _get_next_batch_sync(self) -> Optional[List[Dict[str, str]]]:
        """Synchronous batch retrieval"""
        if self.position >= len(self.data):
            return None
        
        end_pos = min(self.position + self.config.batch_size, len(self.data))
        batch_indices = self.indices[self.position:end_pos]
        batch = [self.data[idx] for idx in batch_indices]
        
        self.position = end_pos
        return batch
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Try to get from prefetch queue first
        try:
            batch = self.prefetch_queue.get_nowait()
            return self._format_batch(batch)
        except:
            pass
        
        # Fallback to direct loading
        batch = self._get_next_batch_sync()
        if batch is None:
            raise StopIteration
        
        return self._format_batch(batch)
    
    def _format_batch(self, batch: List[Dict[str, str]]) -> Dict[str, List[str]]:
        """Format batch for training"""
        return {
            'input_ids': [item['instruction'] for item in batch],
            'ingredients': [item['ingredients'] for item in batch],
            'titles': [item['title'] for item in batch]
        }
    
    def reset(self):
        """Reset for new epoch"""
        self.position = 0
        self.current_epoch += 1
        
        if self.config.shuffle:
            np.random.shuffle(self.indices)
        
        # Clear prefetch queue
        while not self.prefetch_queue.empty():
            try:
                self.prefetch_queue.get_nowait()
            except:
                break
    
    def __len__(self):
        return (len(self.data) + self.config.batch_size - 1) // self.config.batch_size
    
    def get_stats(self) -> Dict[str, Any]:
        return {
            'total_samples': len(self.data),
            'batch_size': self.config.batch_size,
            'current_epoch': self.current_epoch,
            'position': self.position,
            'queue_size': self.prefetch_queue.qsize(),
            'using_rust': False
        }

class FastDataLoader:
    """High-performance data loader with Rust backend and Python fallback"""
    
    def __init__(self, data_path: str, config: Optional[DataLoaderConfig] = None):
        self.data_path = data_path
        self.config = config or DataLoaderConfig()
        
        if RUST_AVAILABLE and self.config.use_rust:
            print("🦀 Using Rust-powered data loader for maximum performance!")
            try:
                self.loader = create_fast_dataloader(
                    data_path, 
                    self.config.batch_size, 
                    self.config.shuffle,
                    self.config.buffer_size
                )
                self.using_rust = True
            except Exception as e:
                print(f"Rust loader failed, falling back to Python: {e}")
                self.loader = PythonFastDataLoader(data_path, self.config)
                self.using_rust = False
        else:
            print("🐍 Using optimized Python data loader")
            self.loader = PythonFastDataLoader(data_path, self.config)
            self.using_rust = False
    
    def __iter__(self):
        return self.loader.__iter__()
    
    def __next__(self):
        return self.loader.__next__()
    
    def reset(self):
        return self.loader.reset()
    
    def __len__(self):
        return len(self.loader)
    
    def get_stats(self) -> Dict[str, Any]:
        stats = self.loader.get_stats()
        stats['using_rust'] = self.using_rust
        return stats

def benchmark_dataloader(data_path: str, num_batches: int = 100, batch_size: int = 8) -> Dict[str, float]:
    """Benchmark both Rust and Python data loaders"""
    results = {}
    
    # Test Rust version if available
    if RUST_AVAILABLE:
        print("🦀 Benchmarking Rust data loader...")
        try:
            start_time = time.time()
            rust_time = benchmark_loading(data_path, num_batches, batch_size)
            results['rust_total_time'] = rust_time
            results['rust_samples_per_sec'] = (num_batches * batch_size) / rust_time
            results['rust_avg_batch_time'] = rust_time / num_batches * 1000  # ms
            print(f"Rust: {results['rust_samples_per_sec']:.1f} samples/sec, {results['rust_avg_batch_time']:.1f}ms/batch")
        except Exception as e:
            print(f"Rust benchmark failed: {e}")
            results['rust_error'] = str(e)
    
    # Test Python version
    print("🐍 Benchmarking Python data loader...")
    try:
        config = DataLoaderConfig(batch_size=batch_size, use_rust=False)
        loader = PythonFastDataLoader(data_path, config)
        
        start_time = time.time()
        for i, batch in enumerate(loader):
            if i >= num_batches:
                break
        
        python_time = time.time() - start_time
        results['python_total_time'] = python_time
        results['python_samples_per_sec'] = (num_batches * batch_size) / python_time
        results['python_avg_batch_time'] = python_time / num_batches * 1000  # ms
        print(f"Python: {results['python_samples_per_sec']:.1f} samples/sec, {results['python_avg_batch_time']:.1f}ms/batch")
        
    except Exception as e:
        print(f"Python benchmark failed: {e}")
        results['python_error'] = str(e)
    
    # Calculate speedup
    if 'rust_samples_per_sec' in results and 'python_samples_per_sec' in results:
        results['speedup'] = results['rust_samples_per_sec'] / results['python_samples_per_sec']
        print(f"🚀 Rust is {results['speedup']:.1f}x faster than Python!")
    
    return results

if __name__ == "__main__":
    # Demo usage
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
        print(f"Benchmarking data loader with {data_path}")
        results = benchmark_dataloader(data_path, num_batches=50, batch_size=8)
        print("\nBenchmark Results:")
        for key, value in results.items():
            print(f"  {key}: {value}")
    else:
        print("Usage: python fast_dataloader.py <path_to_training_data>")
        print("Example: python fast_dataloader.py data/training.json")