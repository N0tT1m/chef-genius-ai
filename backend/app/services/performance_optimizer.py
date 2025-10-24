"""
Performance Optimization Service for Chef Genius

This module provides various performance optimizations including model quantization,
caching, batch processing, and memory management for AI models.
"""

import logging
import time
import asyncio
from typing import Dict, Any, List, Optional, Tuple, Union
from functools import wraps
import hashlib
import pickle
import json
from datetime import datetime, timedelta
import torch
import gc
from collections import OrderedDict
import psutil
from transformers import BitsAndBytesConfig
import numpy as np

logger = logging.getLogger(__name__)

class ModelOptimizer:
    """Handles model quantization and optimization."""
    
    @staticmethod
    def get_quantization_config(bits: int = 4, compute_dtype: torch.dtype = torch.float16) -> BitsAndBytesConfig:
        """Get quantization configuration for model loading."""
        if bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif bits == 8:
            return BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=compute_dtype,
            )
        else:
            raise ValueError("Only 4-bit and 8-bit quantization supported")
    
    @staticmethod
    def optimize_model_for_inference(model):
        """Apply various optimizations for inference."""
        try:
            # Enable inference mode
            model.eval()
            
            # Compile model if using PyTorch 2.0+
            if hasattr(torch, 'compile'):
                try:
                    model = torch.compile(model, mode="reduce-overhead")
                    logger.info("Model compiled with PyTorch 2.0 compiler")
                except Exception as e:
                    logger.warning(f"Failed to compile model: {e}")
            
            # Enable memory efficient attention if available
            if hasattr(model.config, 'use_cache'):
                model.config.use_cache = True
            
            return model
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
            return model
    
    @staticmethod
    def get_model_memory_usage(model) -> Dict[str, float]:
        """Get memory usage statistics for a model."""
        try:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                gpu_reserved = torch.cuda.memory_reserved() / 1024**3  # GB
            else:
                gpu_memory = 0
                gpu_reserved = 0
            
            # Get model parameter count
            param_count = sum(p.numel() for p in model.parameters())
            param_memory = param_count * 4 / 1024**3  # Assuming float32, in GB
            
            return {
                "gpu_allocated_gb": gpu_memory,
                "gpu_reserved_gb": gpu_reserved,
                "model_parameters": param_count,
                "estimated_param_memory_gb": param_memory
            }
            
        except Exception as e:
            logger.error(f"Failed to get memory usage: {e}")
            return {}

class IntelligentCache:
    """Advanced caching system with TTL, LRU eviction, and semantic similarity."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        """
        Initialize the cache.
        
        Args:
            max_size: Maximum number of cached items
            default_ttl: Default time-to-live in seconds
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = OrderedDict()
        self.access_times = {}
        self.ttl_times = {}
        
    def _generate_key(self, data: Any) -> str:
        """Generate a hash key for the data."""
        if isinstance(data, dict):
            # Sort dict items for consistent hashing
            sorted_data = json.dumps(data, sort_keys=True)
        elif isinstance(data, (list, tuple)):
            sorted_data = json.dumps(sorted(data) if all(isinstance(x, (str, int, float)) for x in data) else list(data))
        else:
            sorted_data = str(data)
        
        return hashlib.md5(sorted_data.encode()).hexdigest()
    
    def _is_expired(self, key: str) -> bool:
        """Check if a cache entry has expired."""
        if key not in self.ttl_times:
            return True
        return datetime.now() > self.ttl_times[key]
    
    def _evict_expired(self):
        """Remove expired entries."""
        now = datetime.now()
        expired_keys = [
            key for key, expiry_time in self.ttl_times.items()
            if now > expiry_time
        ]
        
        for key in expired_keys:
            self._remove_entry(key)
    
    def _remove_entry(self, key: str):
        """Remove an entry from all cache structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.ttl_times.pop(key, None)
    
    def _evict_lru(self):
        """Evict least recently used items if cache is full."""
        while len(self.cache) >= self.max_size:
            # Remove least recently used item
            lru_key = min(self.access_times.items(), key=lambda x: x[1])[0]
            self._remove_entry(lru_key)
    
    def get(self, key_data: Any) -> Optional[Any]:
        """Get item from cache."""
        key = self._generate_key(key_data)
        
        # Clean expired entries
        self._evict_expired()
        
        if key in self.cache and not self._is_expired(key):
            # Update access time
            self.access_times[key] = time.time()
            # Move to end (most recent)
            self.cache.move_to_end(key)
            return self.cache[key]
        
        return None
    
    def put(self, key_data: Any, value: Any, ttl: Optional[int] = None) -> None:
        """Put item in cache."""
        key = self._generate_key(key_data)
        
        # Clean expired entries
        self._evict_expired()
        
        # Evict LRU if necessary
        self._evict_lru()
        
        # Add new entry
        self.cache[key] = value
        self.access_times[key] = time.time()
        
        # Set TTL
        ttl = ttl or self.default_ttl
        self.ttl_times[key] = datetime.now() + timedelta(seconds=ttl)
    
    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.access_times.clear()
        self.ttl_times.clear()
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        self._evict_expired()
        
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "utilization": len(self.cache) / self.max_size,
            "default_ttl": self.default_ttl
        }

class BatchProcessor:
    """Handles batch processing for improved throughput."""
    
    def __init__(self, batch_size: int = 8, max_wait_time: float = 0.1):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Maximum batch size
            max_wait_time: Maximum time to wait for batch in seconds
        """
        self.batch_size = batch_size
        self.max_wait_time = max_wait_time
        self.pending_requests = []
        self.request_futures = []
        self.last_batch_time = time.time()
    
    async def add_request(self, request_data: Any, process_func) -> Any:
        """Add a request to the batch queue."""
        future = asyncio.Future()
        
        self.pending_requests.append(request_data)
        self.request_futures.append(future)
        
        # Process batch if full or timeout reached
        if (len(self.pending_requests) >= self.batch_size or 
            time.time() - self.last_batch_time > self.max_wait_time):
            await self._process_batch(process_func)
        
        return await future
    
    async def _process_batch(self, process_func):
        """Process the current batch of requests."""
        if not self.pending_requests:
            return
        
        try:
            # Process batch
            results = await process_func(self.pending_requests)
            
            # Return results to futures
            for future, result in zip(self.request_futures, results):
                if not future.done():
                    future.set_result(result)
            
        except Exception as e:
            # Set exception for all futures
            for future in self.request_futures:
                if not future.done():
                    future.set_exception(e)
        
        finally:
            # Clear batch
            self.pending_requests.clear()
            self.request_futures.clear()
            self.last_batch_time = time.time()

class MemoryManager:
    """Manages memory usage and performs cleanup."""
    
    @staticmethod
    def get_memory_stats() -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        stats = {
            "ram_used_gb": memory_info.rss / 1024**3,
            "ram_available_gb": psutil.virtual_memory().available / 1024**3,
            "ram_percent": psutil.virtual_memory().percent
        }
        
        if torch.cuda.is_available():
            stats.update({
                "gpu_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "gpu_reserved_gb": torch.cuda.memory_reserved() / 1024**3,
                "gpu_free_gb": (torch.cuda.get_device_properties(0).total_memory - 
                               torch.cuda.memory_allocated()) / 1024**3
            })
        
        return stats
    
    @staticmethod
    def cleanup_memory():
        """Perform memory cleanup."""
        try:
            # Python garbage collection
            gc.collect()
            
            # PyTorch memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            logger.info("Memory cleanup completed")
            
        except Exception as e:
            logger.error(f"Memory cleanup failed: {e}")
    
    @staticmethod
    def should_cleanup_memory(threshold_percent: float = 80.0) -> bool:
        """Check if memory cleanup should be performed."""
        stats = MemoryManager.get_memory_stats()
        
        # Check RAM usage
        if stats["ram_percent"] > threshold_percent:
            return True
        
        # Check GPU memory usage
        if torch.cuda.is_available():
            gpu_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            gpu_usage_percent = (stats["gpu_allocated_gb"] / gpu_total) * 100
            if gpu_usage_percent > threshold_percent:
                return True
        
        return False

class PerformanceMonitor:
    """Monitors and logs performance metrics."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation."""
        self.start_times[operation] = time.time()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration."""
        if operation not in self.start_times:
            logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        
        # Store metric
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append({
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep only last 100 measurements
        if len(self.metrics[operation]) > 100:
            self.metrics[operation] = self.metrics[operation][-100:]
        
        del self.start_times[operation]
        return duration
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get performance statistics for an operation."""
        if operation not in self.metrics or not self.metrics[operation]:
            return {}
        
        durations = [m["duration"] for m in self.metrics[operation]]
        
        return {
            "count": len(durations),
            "avg_duration": np.mean(durations),
            "min_duration": np.min(durations),
            "max_duration": np.max(durations),
            "std_duration": np.std(durations),
            "p95_duration": np.percentile(durations, 95),
            "p99_duration": np.percentile(durations, 99)
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Get performance statistics for all operations."""
        return {op: self.get_stats(op) for op in self.metrics.keys()}

def performance_monitor(operation_name: str):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            monitor = getattr(func, '_performance_monitor', None)
            if not monitor:
                monitor = PerformanceMonitor()
                func._performance_monitor = monitor
            
            monitor.start_timer(operation_name)
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                duration = monitor.end_timer(operation_name)
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            monitor = getattr(func, '_performance_monitor', None)
            if not monitor:
                monitor = PerformanceMonitor()
                func._performance_monitor = monitor
            
            monitor.start_timer(operation_name)
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                duration = monitor.end_timer(operation_name)
                logger.debug(f"{operation_name} completed in {duration:.3f}s")
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self):
        self.cache = IntelligentCache(max_size=2000, default_ttl=1800)  # 30 min TTL
        self.batch_processor = BatchProcessor(batch_size=8, max_wait_time=0.1)
        self.memory_manager = MemoryManager()
        self.performance_monitor = PerformanceMonitor()
        
        # Auto cleanup settings
        self.auto_cleanup_enabled = True
        self.cleanup_interval = 300  # 5 minutes
        self.last_cleanup = time.time()
    
    async def optimize_model_loading(self, model_name: str, quantization_bits: int = 4) -> Dict[str, Any]:
        """Provide optimized model loading configuration."""
        config = {
            "model_name": model_name,
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
            "trust_remote_code": True
        }
        
        # Add quantization if CUDA available
        if torch.cuda.is_available() and quantization_bits in [4, 8]:
            try:
                config["quantization_config"] = ModelOptimizer.get_quantization_config(
                    bits=quantization_bits,
                    compute_dtype=torch.float16
                )
                logger.info(f"Enabled {quantization_bits}-bit quantization")
            except Exception as e:
                logger.warning(f"Failed to enable quantization: {e}")
        
        return config
    
    def should_auto_cleanup(self) -> bool:
        """Check if automatic cleanup should be performed."""
        if not self.auto_cleanup_enabled:
            return False
        
        # Time-based cleanup
        if time.time() - self.last_cleanup > self.cleanup_interval:
            return True
        
        # Memory-based cleanup
        return self.memory_manager.should_cleanup_memory()
    
    async def auto_cleanup_if_needed(self):
        """Perform automatic cleanup if conditions are met."""
        if self.should_auto_cleanup():
            await self.cleanup_resources()
    
    async def cleanup_resources(self):
        """Perform comprehensive resource cleanup."""
        try:
            logger.info("Starting resource cleanup...")
            
            # Clear caches
            self.cache.clear()
            
            # Memory cleanup
            self.memory_manager.cleanup_memory()
            
            # Update cleanup time
            self.last_cleanup = time.time()
            
            logger.info("Resource cleanup completed")
            
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get comprehensive system performance statistics."""
        return {
            "memory": self.memory_manager.get_memory_stats(),
            "cache": self.cache.stats(),
            "performance": self.performance_monitor.get_all_stats(),
            "auto_cleanup_enabled": self.auto_cleanup_enabled,
            "last_cleanup": self.last_cleanup
        }