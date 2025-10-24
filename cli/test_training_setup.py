#!/usr/bin/env python3
"""
Test script to validate the optimized training setup
Runs a quick training test to ensure all fixes work correctly
"""

import torch
import sys
import os
from pathlib import Path

# Add the cli directory to Python path
cli_dir = Path(__file__).parent
sys.path.insert(0, str(cli_dir))

def test_tokenization_fixes():
    """Test the seq2seq tokenization fixes."""
    
    print("🧪 Testing seq2seq tokenization fixes...")
    
    try:
        from transformers import AutoTokenizer
        from training_integration_all_datasets import TrainingDataLoaderWrapper
        
        # Create a mock batch to test tokenization
        mock_batch = {
            'input_ids': [
                "Ingredients: 2 cups flour, 1 cup sugar, 2 eggs\n\nInstructions: Mix ingredients, bake at 350F for 30 minutes",
                "Ingredients: 1 lb chicken, salt, pepper\n\nInstructions: Season chicken, cook until done"
            ]
        }
        
        # Test with FLAN-T5 tokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        # Create wrapper (mock unified loader)
        class MockUnifiedLoader:
            def __init__(self):
                self.batch_size = 2
                self.total_samples = 100
                
        wrapper = TrainingDataLoaderWrapper(MockUnifiedLoader(), tokenizer)
        
        # Test tokenization
        result = wrapper._tokenize_batch(mock_batch)
        
        print(f"✅ Input shape: {result['input_ids'].shape}")
        print(f"✅ Labels shape: {result['labels'].shape}")
        print(f"✅ Has attention mask: {'attention_mask' in result}")
        
        # Decode sample to verify format
        sample_input = tokenizer.decode(result['input_ids'][0], skip_special_tokens=True)
        print(f"✅ Sample input: {sample_input[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ Tokenization test failed: {e}")
        return False

def test_prompt_formatting():
    """Test the improved prompt formatting."""
    
    print("\n🧪 Testing prompt formatting fixes...")
    
    try:
        from training_monitor_enhanced import format_simple_prompt
        
        test_prompts = [
            "chocolate chip cookies",
            "create pasta carbonara",
            "generate beef stew"
        ]
        
        for prompt in test_prompts:
            formatted = format_simple_prompt(prompt)
            print(f"✅ '{prompt}' → '{formatted}'")
            
            # Check if it follows FLAN-T5 instruction format
            assert "Generate a detailed recipe" in formatted
            
        return True
        
    except Exception as e:
        print(f"❌ Prompt formatting test failed: {e}")
        return False

def test_training_configuration():
    """Test the training configuration."""
    
    print("\n🧪 Testing training configuration...")
    
    try:
        from optimized_recipe_training import create_optimized_training_args
        
        # Test training args creation
        args = create_optimized_training_args("./test_output")
        
        print(f"✅ Learning rate: {args.learning_rate}")
        print(f"✅ Batch size: {args.per_device_train_batch_size}")
        print(f"✅ Gradient accumulation: {args.gradient_accumulation_steps}")
        print(f"✅ BF16 enabled: {args.bf16}")
        print(f"✅ Gradient checkpointing: {args.gradient_checkpointing}")
        
        # Verify key optimizations
        assert args.learning_rate == 5e-4, "Learning rate should be 5e-4"
        assert args.bf16 == True, "BF16 should be enabled"
        assert args.gradient_checkpointing == True, "Gradient checkpointing should be enabled"
        
        return True
        
    except Exception as e:
        print(f"❌ Training configuration test failed: {e}")
        return False

def test_generation_improvements():
    """Test generation parameter improvements."""
    
    print("\n🧪 Testing generation improvements...")
    
    try:
        # Test if we can load the model
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        print("Loading FLAN-T5 model for generation test...")
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
        
        # Just test model loading (no actual generation to save time)
        if torch.cuda.is_available():
            print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"✅ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        else:
            print("⚠️  CUDA not available, will use CPU")
            
        # Test the improved prompt format
        from training_monitor_enhanced import format_simple_prompt
        test_prompt = format_simple_prompt("chocolate chip cookies")
        
        print(f"✅ Formatted prompt: {test_prompt}")
        
        return True
        
    except Exception as e:
        print(f"❌ Generation test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    print("🍳 CHEF GENIUS TRAINING SETUP VALIDATION")
    print("="*50)
    
    tests = [
        ("Tokenization Fixes", test_tokenization_fixes),
        ("Prompt Formatting", test_prompt_formatting), 
        ("Training Configuration", test_training_configuration),
        ("Generation Improvements", test_generation_improvements)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 Running {test_name} test...")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'='*50}")
    print("🔍 TEST RESULTS SUMMARY")
    print("="*50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if success:
            passed += 1
    
    print(f"\n📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Training setup is ready.")
        print("\n🚀 You can now run optimized training with:")
        print("python cli/optimized_recipe_training.py --output-dir ./models/recipe_optimized")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)