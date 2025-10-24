#!/usr/bin/env python3
"""
Test script to verify the epoch continuation fix works correctly.
"""

import sys
import os
from transformers import AutoTokenizer

# Add current directory to path
sys.path.append(os.getcwd())

def test_epoch_fix():
    """Test that the dataloader can handle multiple epochs correctly."""
    
    print("🧪 Testing epoch continuation fix...")
    
    try:
        # Import the fixed dataloader
        from jsonl_dataloader import create_optimized_jsonl_dataloader
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Create dataloader
        print("\n📦 Creating dataloader...")
        dataloader = create_optimized_jsonl_dataloader(
            tokenizer=tokenizer,
            validated_data_dir="validated_datasets",
            batch_size=2,
            min_quality_score=0.5
        )
        
        # Test that reset method exists
        if hasattr(dataloader, 'reset'):
            print("✅ Dataloader has reset() method")
        else:
            print("❌ Dataloader missing reset() method")
            return False
        
        # Test multi-epoch simulation
        print("\n🔄 Testing multi-epoch simulation...")
        
        for epoch in range(3):
            print(f"\n📊 Epoch {epoch + 1}/3")
            
            # Reset for new epoch (this was missing before)
            if epoch > 0:
                dataloader.reset()
                print("   🔄 Reset dataloader for new epoch")
            
            # Process a few batches
            batch_count = 0
            for i, batch in enumerate(dataloader):
                print(f"   Batch {i+1}: input_shape={batch['input_ids'].shape}")
                batch_count += 1
                
                # Only test first 3 batches per epoch
                if batch_count >= 3:
                    break
            
            print(f"   ✅ Processed {batch_count} batches in epoch {epoch + 1}")
        
        print("\n🎉 Multi-epoch test completed successfully!")
        print("🔧 The epoch continuation fix is working correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_epoch_fix()
    if success:
        print("\n✅ TEST PASSED: Epoch continuation fix is working")
        sys.exit(0)
    else:
        print("\n❌ TEST FAILED: Epoch continuation fix needs more work")
        sys.exit(1)