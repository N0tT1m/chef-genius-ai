#!/usr/bin/env python3
"""
Memory-Optimized Training Pipeline for OOM Prevention
Solves data pipeline bottleneck by using Rust dataloader with smart memory management
"""

import torch
import gc
from typing import Dict, Optional
import os

class FakeDataset:
    """Fake dataset to provide len() for trainer compatibility."""
    def __init__(self, total_samples):
        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples


class MemoryOptimizedDataloader:
    """
    Wrapper around Rust dataloader that prevents OOM by:
    1. Pre-tokenizing in smaller chunks
    2. Using gradient accumulation effectively
    3. Clearing cache strategically
    """

    def __init__(self, rust_dataloader, tokenizer, max_input_length=256, max_target_length=512):
        self.rust_dataloader = rust_dataloader
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length

        # Memory management settings
        self.cache_clear_frequency = 10  # Clear every N batches
        self.batch_counter = 0

        # Cache pad token ID
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_token_id = self.tokenizer.pad_token_id

        # Add fake dataset attribute for trainer compatibility
        try:
            stats = self.rust_dataloader.get_stats()
            total_samples = stats.get('total_samples', 2490151)  # Use actual or fallback
        except:
            total_samples = 2490151  # Fallback estimate

        self.dataset = FakeDataset(total_samples)

    def __iter__(self):
        self.batch_counter = 0
        return self

    def __next__(self):
        # Get batch from Rust (super fast)
        rust_batch = next(self.rust_dataloader)

        # Tokenize with memory optimization
        batch = self._tokenize_with_memory_management(rust_batch)

        # Periodic memory cleanup
        self.batch_counter += 1
        if self.batch_counter % self.cache_clear_frequency == 0:
            torch.cuda.empty_cache()
            gc.collect()

        return batch

    def _tokenize_with_memory_management(self, rust_batch):
        """Tokenize with minimal memory footprint"""
        inputs = rust_batch.get('input_ids', [])
        outputs = rust_batch.get('outputs', [])
        quality_scores = rust_batch.get('quality_scores', [])

        # Tokenize in single batch (most efficient)
        # Use 'longest' padding instead of 'max_length' to save memory
        input_encodings = self.tokenizer(
            inputs,
            truncation=True,
            padding='longest',  # Only pad to longest in batch, not max_length
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        target_encodings = self.tokenizer(
            outputs,
            truncation=True,
            padding='longest',  # Only pad to longest in batch, not max_length
            max_length=self.max_target_length,
            return_tensors='pt'
        )

        # Prepare labels (replace pad tokens with -100)
        labels = target_encodings['input_ids'].clone()
        labels[labels == self.pad_token_id] = -100

        # Clean up intermediate tensors immediately
        del target_encodings

        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': labels,
            'quality_scores': torch.tensor(quality_scores, dtype=torch.float32) if quality_scores else None
        }

    def __len__(self):
        try:
            return len(self.rust_dataloader)
        except:
            # Fallback: estimate from dataset
            return len(self.dataset)

    def reset(self):
        """Reset for new epoch"""
        self.rust_dataloader.reset()
        self.batch_counter = 0
        torch.cuda.empty_cache()
        gc.collect()


class AdaptiveBatchSizer:
    """
    Dynamically adjusts batch size based on available GPU memory.
    Prevents OOM while maximizing throughput.
    """

    def __init__(self, initial_batch_size=4, gradient_accumulation_steps=32):
        self.batch_size = initial_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.oom_count = 0

        # Get GPU memory stats
        if torch.cuda.is_available():
            self.total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            self.is_5090 = self.total_memory > 30  # RTX 5090 has 32GB
            print(f"📊 Detected GPU with {self.total_memory:.1f}GB VRAM")

            if self.is_5090:
                print("🎯 RTX 5090 detected - using aggressive memory optimization")
        else:
            self.total_memory = 0
            self.is_5090 = False

    def get_effective_batch_size(self):
        """Get effective batch size (batch_size * gradient_accumulation_steps)"""
        return self.batch_size * self.gradient_accumulation_steps

    def handle_oom(self):
        """Called when OOM occurs - reduces batch size"""
        self.oom_count += 1

        if self.batch_size > 1:
            self.batch_size = max(1, self.batch_size // 2)
            # Increase gradient accumulation to maintain effective batch size
            self.gradient_accumulation_steps *= 2

            torch.cuda.empty_cache()
            gc.collect()

            print(f"⚠️  OOM #{self.oom_count}: Reduced batch size to {self.batch_size}, increased grad accum to {self.gradient_accumulation_steps}")
            print(f"   Effective batch size: {self.get_effective_batch_size()}")
            return True
        else:
            print(f"❌ Cannot reduce batch size further (already at 1)")
            return False

    def get_recommended_settings(self, model_size_gb: float) -> Dict:
        """Get recommended batch size based on model size and GPU memory"""
        available_memory = self.total_memory * 0.7  # Use 70% of total memory safely
        memory_for_activations = available_memory - model_size_gb

        # Estimate: each sample in batch uses ~0.5GB for FLAN-T5-XL activations
        sample_memory_estimate = 0.5  # GB per sample

        recommended_batch = max(1, int(memory_for_activations / sample_memory_estimate))

        # For RTX 5090 (32GB), we can be more aggressive
        if self.is_5090:
            # FLAN-T5-XL (3B) on RTX 5090 can handle larger batches
            if model_size_gb < 6:  # XL model
                recommended_batch = min(8, recommended_batch)
                recommended_grad_accum = 16
            elif model_size_gb < 15:  # XXL model
                recommended_batch = min(4, recommended_batch)
                recommended_grad_accum = 32
            else:
                recommended_batch = max(2, recommended_batch)
                recommended_grad_accum = 64
        else:
            # RTX 4090 (24GB) - more conservative
            recommended_batch = min(4, recommended_batch)
            recommended_grad_accum = 32

        return {
            'batch_size': recommended_batch,
            'gradient_accumulation_steps': recommended_grad_accum,
            'effective_batch_size': recommended_batch * recommended_grad_accum,
            'estimated_memory_usage': model_size_gb + (recommended_batch * sample_memory_estimate)
        }


def create_memory_optimized_dataloader(tokenizer, validated_data_dir="cli/validated_datasets",
                                      batch_size=4, min_quality_score=0.6):
    """
    Create a memory-optimized Rust dataloader that prevents OOM.

    Key optimizations:
    1. Uses Rust backend for ultra-fast data loading (100-1000x faster)
    2. Uses 'longest' padding instead of 'max_length' to reduce memory
    3. Periodic cache clearing to prevent memory buildup
    4. Aggressive garbage collection
    """

    print("🦀 Creating memory-optimized Rust dataloader...")

    # Try to use Rust dataloader
    try:
        from chef_genius_dataloader import create_fast_dataloader
        from pathlib import Path

        # Find combined JSONL file
        data_path = Path(validated_data_dir)
        combined_file = data_path / "combined_all_datasets_flan_t5.jsonl"

        if not combined_file.exists():
            raise FileNotFoundError(f"Combined JSONL file not found: {combined_file}")

        # Create Rust dataloader
        rust_loader = create_fast_dataloader(
            str(combined_file),
            batch_size,
            True,  # shuffle
            16  # buffer_size - smaller to reduce memory
        )

        # Wrap with memory optimization
        # Reduced sequence lengths for OOM prevention with large dataset (6.6M recipes)
        optimized_loader = MemoryOptimizedDataloader(
            rust_loader,
            tokenizer,
            max_input_length=196,  # Reduced from 256
            max_target_length=384   # Reduced from 512
        )

        print(f"✅ Memory-optimized Rust dataloader created!")
        print(f"   Batch size: {batch_size}")
        print(f"   Cache clearing: Every 10 batches")
        print(f"   Padding strategy: 'longest' (memory efficient)")

        return optimized_loader

    except ImportError:
        print("❌ Rust dataloader not available - falling back to Python")
        # Fallback to Python dataloader with same optimizations
        from jsonl_dataloader import create_optimized_jsonl_dataloader
        return create_optimized_jsonl_dataloader(
            tokenizer=tokenizer,
            validated_data_dir=validated_data_dir,
            batch_size=batch_size,
            min_quality_score=min_quality_score
        )


def optimize_model_for_memory(model):
    """
    Apply aggressive memory optimizations to model.
    """

    print("🔧 Applying aggressive memory optimizations...")

    # 1. Enable gradient checkpointing (trades compute for memory)
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("   ✅ Gradient checkpointing enabled")

    # 2. Convert to bfloat16 (saves 50% memory vs float32)
    if torch.cuda.is_available():
        model = model.to(dtype=torch.bfloat16)
        print("   ✅ Model converted to bfloat16")

    # 3. Optimize CUDA memory allocator
    if torch.cuda.is_available():
        # Aggressive memory fragmentation settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128,roundup_power2_divisions:16'
        print("   ✅ CUDA memory allocator optimized")

        # Clear cache
        torch.cuda.empty_cache()
        gc.collect()

    # 4. Enable Flash Attention if available (30% faster + less memory)
    try:
        if hasattr(model, 'enable_flash_attention'):
            model.enable_flash_attention()
            print("   ✅ Flash Attention enabled")
    except:
        pass

    return model


def print_memory_stats():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024**3)
        reserved = torch.cuda.memory_reserved() / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)

        print(f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved, {total:.2f}GB total")
        print(f"   Usage: {(allocated/total)*100:.1f}%")
    else:
        print("GPU not available")


if __name__ == "__main__":
    print("Memory-Optimized Training Components")
    print("=" * 60)

    # Test adaptive batch sizer
    print("\n📊 Testing Adaptive Batch Sizer...")
    sizer = AdaptiveBatchSizer(initial_batch_size=4, gradient_accumulation_steps=32)

    # Simulate FLAN-T5-XL (3B params, ~6GB)
    model_size = 6.0
    recommendations = sizer.get_recommended_settings(model_size)

    print(f"\n🎯 Recommendations for {model_size}GB model:")
    print(f"   Batch size: {recommendations['batch_size']}")
    print(f"   Gradient accumulation: {recommendations['gradient_accumulation_steps']}")
    print(f"   Effective batch size: {recommendations['effective_batch_size']}")
    print(f"   Estimated memory usage: {recommendations['estimated_memory_usage']:.1f}GB")

    # Show current GPU memory
    print("\n💾 Current GPU Memory:")
    print_memory_stats()
