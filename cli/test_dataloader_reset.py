#!/usr/bin/env python3
"""
Test script to verify data loader reset functionality
"""

import time
import threading
from unified_dataset_loader import UnifiedDatasetLoader

def test_dataloader_reset():
    """Test that data loader properly resets between epochs."""
    
    print("🧪 Testing Data Loader Reset Functionality")
    print("=" * 60)
    
    # Create unified loader
    loader = UnifiedDatasetLoader()
    
    # Test with 3 simulated epochs
    for epoch in range(3):
        print(f"\n🚀 Epoch {epoch + 1}")
        print("-" * 30)
        
        # Reset before each epoch (except first)
        if epoch > 0:
            print("🔄 Resetting data loader...")
            loader.reset()
            time.sleep(1)  # Give reset time to complete
        
        # Test iteration
        batch_count = 0
        start_time = time.time()
        
        try:
            # Create iterator
            data_iter = iter(loader)
            
            # Try to get a few batches
            for i in range(5):  # Just test first 5 batches
                try:
                    batch = next(data_iter)
                    batch_count += 1
                    print(f"  ✅ Batch {batch_count}: Got {len(batch.get('input_ids', []))} samples")
                except StopIteration:
                    print(f"  ⚠️  Data exhausted after {batch_count} batches")
                    break
                except Exception as e:
                    print(f"  ❌ Error getting batch: {e}")
                    break
            
            elapsed = time.time() - start_time
            print(f"  ⏱️  Epoch {epoch + 1} processed {batch_count} batches in {elapsed:.2f}s")
            
            # Check thread status
            stats = loader.get_stats()
            print(f"  📊 Active threads: {stats.get('active_threads', 0)}")
            print(f"  📊 Buffer size: {stats.get('buffer_size', 0)}")
            
        except Exception as e:
            print(f"  💥 Epoch {epoch + 1} failed: {e}")
    
    print(f"\n🏁 Reset test complete!")

if __name__ == "__main__":
    test_dataloader_reset()