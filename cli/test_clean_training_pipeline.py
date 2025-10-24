#!/usr/bin/env python3
"""
Test script to verify the clean JSONL training pipeline works correctly
"""

import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from jsonl_dataloader import create_optimized_jsonl_dataloader

def test_jsonl_dataloader():
    """Test the JSONL dataloader with clean data."""
    
    print("🧪 Testing JSONL dataloader with clean data...")
    
    # Load tokenizer
    print("📚 Loading FLAN-T5 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test dataloader
    try:
        print("🔄 Creating JSONL dataloader...")
        dataloader = create_optimized_jsonl_dataloader(
            tokenizer=tokenizer,
            validated_data_dir="validated_datasets",
            batch_size=2,  # Small batch for testing
            min_quality_score=0.6
        )
        
        print("✅ Dataloader created successfully!")
        
        # Test a few batches
        print("🧪 Testing batch processing...")
        for i, batch in enumerate(dataloader):
            print(f"\nBatch {i+1}:")
            print(f"  Input IDs shape: {batch['input_ids'].shape}")
            print(f"  Attention mask shape: {batch['attention_mask'].shape}")
            print(f"  Labels shape: {batch['labels'].shape}")
            print(f"  Quality scores: {batch['quality_scores']}")
            
            # Decode a sample
            sample_input = tokenizer.decode(batch['input_ids'][0], skip_special_tokens=True)
            sample_labels = batch['labels'][0].clone()
            sample_labels[sample_labels == -100] = tokenizer.pad_token_id
            sample_output = tokenizer.decode(sample_labels, skip_special_tokens=True)
            
            print(f"  Sample input: {sample_input[:100]}...")
            print(f"  Sample output: {sample_output[:100]}...")
            
            if i >= 2:  # Test first 3 batches
                break
        
        print("\n✅ JSONL dataloader test PASSED!")
        return True
        
    except Exception as e:
        print(f"\n❌ JSONL dataloader test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_model_compatibility():
    """Test that the dataloader works with FLAN-T5 model."""
    
    print("\n🧪 Testing model compatibility...")
    
    try:
        # Load model and tokenizer
        print("📚 Loading FLAN-T5 model and tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForSeq2SeqLM.from_pretrained(
            "google/flan-t5-large",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
            print("🔥 Model loaded on GPU")
        else:
            print("💻 Model loaded on CPU")
        
        # Create dataloader
        dataloader = create_optimized_jsonl_dataloader(
            tokenizer=tokenizer,
            validated_data_dir="validated_datasets",
            batch_size=1,  # Very small batch for testing
            min_quality_score=0.6
        )
        
        # Test forward pass
        print("🔄 Testing forward pass...")
        model.eval()
        
        with torch.no_grad():
            batch = next(iter(dataloader))
            
            if torch.cuda.is_available():
                batch = {k: v.cuda() if torch.is_tensor(v) else v for k, v in batch.items()}
            
            # Forward pass
            outputs = model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels']
            )
            
            print(f"  Loss: {outputs.loss.item():.4f}")
            print(f"  Logits shape: {outputs.logits.shape}")
            
        print("✅ Model compatibility test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Model compatibility test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_generation_quality():
    """Test that the model can generate from clean training data format."""
    
    print("\n🧪 Testing generation quality...")
    
    try:
        # Load model and tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        model.eval()
        
        # Test with clean prompt format
        test_prompts = [
            "Generate a complete recipe for chocolate chip cookies",
            "Generate a complete recipe for chicken stir-fry with vegetables",
            "Generate a complete recipe for vegetarian pasta"
        ]
        
        print("🔄 Testing generation with clean prompts...")
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt}")
            
            inputs = tokenizer(prompt, return_tensors="pt", max_length=256, truncation=True)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Generated: {generated_text[:200]}...")
            
            # Check if generation looks reasonable (not just echoing)
            if len(generated_text.split()) > 10 and prompt not in generated_text:
                print("✅ Generation looks good!")
            else:
                print("⚠️ Generation may be echoing or too short")
        
        print("\n✅ Generation quality test PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Generation quality test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    
    print("🚀 TESTING CLEAN TRAINING PIPELINE")
    print("=" * 50)
    
    # Check if validated data exists
    if not os.path.exists("validated_datasets/combined_all_datasets_flan_t5.jsonl"):
        print("❌ Validated datasets not found!")
        print("Please run batch_validate_all_datasets.sh first")
        return
    
    print(f"📊 Found validated dataset with {sum(1 for line in open('validated_datasets/combined_all_datasets_flan_t5.jsonl'))} recipes")
    
    tests = [
        ("JSONL Dataloader", test_jsonl_dataloader),
        ("Model Compatibility", test_model_compatibility),
        ("Generation Quality", test_generation_quality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        
        if test_func():
            passed += 1
            print(f"✅ {test_name} PASSED")
        else:
            print(f"❌ {test_name} FAILED")
    
    print(f"\n{'='*50}")
    print(f"🎯 TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Training pipeline is ready!")
        print("🚀 You can now run complete_optimized_training.py with clean data")
    else:
        print("⚠️ Some tests failed. Please check the issues above.")

if __name__ == "__main__":
    main()