#!/usr/bin/env python3
"""
Compare recipe generation quality between FLAN-T5 and Mistral-7B-Instruct
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, BitsAndBytesConfig
import time
import json
from pathlib import Path
import argparse

class ModelComparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.flan_t5_model = None
        self.flan_t5_tokenizer = None
        self.mistral_model = None
        self.mistral_tokenizer = None
        
    def load_flan_t5(self):
        """Load FLAN-T5 Large model."""
        print("Loading FLAN-T5 Large...")
        model_name = "google/flan-t5-large"
        
        self.flan_t5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.flan_t5_model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        if self.flan_t5_tokenizer.pad_token is None:
            self.flan_t5_tokenizer.pad_token = self.flan_t5_tokenizer.eos_token
            
        print("✅ FLAN-T5 Large loaded")
        
    def load_mistral(self):
        """Load Mistral-7B-Instruct with 4-bit quantization."""
        print("Loading Mistral-7B-Instruct...")
        model_name = "mistralai/Mistral-7B-Instruct-v0.1"
        
        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.mistral_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.mistral_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        if self.mistral_tokenizer.pad_token is None:
            self.mistral_tokenizer.pad_token = self.mistral_tokenizer.eos_token
            self.mistral_tokenizer.pad_token_id = self.mistral_tokenizer.eos_token_id
            
        print("✅ Mistral-7B-Instruct loaded")
        
    def generate_with_flan_t5(self, prompt: str) -> str:
        """Generate recipe with FLAN-T5."""
        # Format prompt for T5
        t5_prompt = f"Generate a recipe using these ingredients: {prompt}"
        
        inputs = self.flan_t5_tokenizer(
            t5_prompt,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        ).to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.flan_t5_model.generate(
                **inputs,
                max_new_tokens=1000,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.flan_t5_tokenizer.pad_token_id,
                eos_token_id=self.flan_t5_tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        generated_text = self.flan_t5_tokenizer.decode(
            outputs[0], 
            skip_special_tokens=True
        )
        
        return generated_text, generation_time
        
    def generate_with_mistral(self, prompt: str) -> str:
        """Generate recipe with Mistral."""
        # Format prompt for Mistral chat template
        messages = [
            {"role": "user", "content": f"Generate a complete recipe using these ingredients: {prompt}. Format it with title, ingredients list, and step-by-step instructions."}
        ]
        
        # Apply chat template
        formatted_prompt = self.mistral_tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        inputs = self.mistral_tokenizer(
            formatted_prompt,
            return_tensors="pt",
            max_length=2048,
            truncation=True,
            padding=True
        ).to(self.device)
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = self.mistral_model.generate(
                **inputs,
                max_new_tokens=1500,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                top_k=50,
                do_sample=True,
                pad_token_id=self.mistral_tokenizer.pad_token_id,
                eos_token_id=self.mistral_tokenizer.eos_token_id
            )
        
        generation_time = time.time() - start_time
        
        # Decode only the generated part
        generated_text = self.mistral_tokenizer.decode(
            outputs[0][len(inputs['input_ids'][0]):], 
            skip_special_tokens=True
        )
        
        return generated_text, generation_time
        
    def compare_models(self, test_prompts: list) -> dict:
        """Compare both models on test prompts."""
        results = {
            "comparisons": [],
            "summary": {
                "flan_t5_avg_time": 0,
                "mistral_avg_time": 0,
                "total_tests": len(test_prompts)
            }
        }
        
        flan_t5_times = []
        mistral_times = []
        
        for i, prompt in enumerate(test_prompts):
            print(f"\n{'='*60}")
            print(f"Test {i+1}/{len(test_prompts)}: {prompt}")
            print('='*60)
            
            # Test FLAN-T5
            try:
                print("🔄 Generating with FLAN-T5...")
                flan_t5_result, flan_t5_time = self.generate_with_flan_t5(prompt)
                flan_t5_times.append(flan_t5_time)
                print(f"⏱️  FLAN-T5 time: {flan_t5_time:.2f}s")
            except Exception as e:
                print(f"❌ FLAN-T5 failed: {e}")
                flan_t5_result = f"Error: {e}"
                flan_t5_time = 0
            
            # Test Mistral
            try:
                print("🔄 Generating with Mistral...")
                mistral_result, mistral_time = self.generate_with_mistral(prompt)
                mistral_times.append(mistral_time)
                print(f"⏱️  Mistral time: {mistral_time:.2f}s")
            except Exception as e:
                print(f"❌ Mistral failed: {e}")
                mistral_result = f"Error: {e}"
                mistral_time = 0
            
            # Store results
            comparison = {
                "prompt": prompt,
                "flan_t5": {
                    "result": flan_t5_result,
                    "time": flan_t5_time,
                    "length": len(flan_t5_result)
                },
                "mistral": {
                    "result": mistral_result,
                    "time": mistral_time,
                    "length": len(mistral_result)
                }
            }
            
            results["comparisons"].append(comparison)
            
            # Print preview
            print(f"\n📝 FLAN-T5 Preview:")
            print(flan_t5_result[:200] + "..." if len(flan_t5_result) > 200 else flan_t5_result)
            print(f"\n📝 Mistral Preview:")
            print(mistral_result[:200] + "..." if len(mistral_result) > 200 else mistral_result)
        
        # Calculate averages
        if flan_t5_times:
            results["summary"]["flan_t5_avg_time"] = sum(flan_t5_times) / len(flan_t5_times)
        if mistral_times:
            results["summary"]["mistral_avg_time"] = sum(mistral_times) / len(mistral_times)
        
        return results

def main():
    """Main comparison function."""
    parser = argparse.ArgumentParser(description="Compare FLAN-T5 vs Mistral for recipe generation")
    parser.add_argument("--output", default="model_comparison.json", help="Output file")
    parser.add_argument("--load-flan-t5", action="store_true", help="Load FLAN-T5 for comparison")
    parser.add_argument("--load-mistral", action="store_true", help="Load Mistral for comparison")
    
    args = parser.parse_args()
    
    if not args.load_flan_t5 and not args.load_mistral:
        print("Please specify --load-flan-t5 and/or --load-mistral")
        return
    
    print("🍳 MODEL COMPARISON: FLAN-T5 vs Mistral-7B-Instruct")
    print("="*60)
    
    # Test prompts
    test_prompts = [
        "chicken, rice, vegetables, soy sauce",
        "pasta, tomatoes, basil, garlic, olive oil",
        "beef, potatoes, carrots, onions",
        "salmon, lemon, herbs, butter",
        "chocolate, eggs, flour, sugar"
    ]
    
    print(f"🧪 Testing with {len(test_prompts)} prompts")
    print(f"📊 GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    comparator = ModelComparator()
    
    # Load models
    if args.load_flan_t5:
        comparator.load_flan_t5()
    
    if args.load_mistral:
        comparator.load_mistral()
    
    # Run comparison
    if args.load_flan_t5 and args.load_mistral:
        results = comparator.compare_models(test_prompts)
        
        # Save results
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        print(f"\n📈 COMPARISON SUMMARY")
        print("="*40)
        print(f"FLAN-T5 avg time: {results['summary']['flan_t5_avg_time']:.2f}s")
        print(f"Mistral avg time: {results['summary']['mistral_avg_time']:.2f}s")
        print(f"Speedup: {results['summary']['flan_t5_avg_time'] / results['summary']['mistral_avg_time']:.2f}x")
        
        print(f"\n✅ Full comparison saved to: {args.output}")
    
    else:
        # Test individual models
        if args.load_flan_t5:
            print("Testing FLAN-T5 only...")
            for prompt in test_prompts[:2]:  # Test first 2
                result, time_taken = comparator.generate_with_flan_t5(prompt)
                print(f"Prompt: {prompt}")
                print(f"Time: {time_taken:.2f}s")
                print(f"Result: {result[:200]}...")
                print("-" * 40)
        
        if args.load_mistral:
            print("Testing Mistral only...")
            for prompt in test_prompts[:2]:  # Test first 2
                result, time_taken = comparator.generate_with_mistral(prompt)
                print(f"Prompt: {prompt}")
                print(f"Time: {time_taken:.2f}s")
                print(f"Result: {result[:200]}...")
                print("-" * 40)

if __name__ == "__main__":
    main()