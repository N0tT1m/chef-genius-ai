#!/usr/bin/env python3
"""
Comprehensive Checkpoint Tester for Enterprise Recipe Generation Model
Tests model checkpoints with edge cases and quality inputs for Discord notifications
"""

import os
import sys
import time
import torch
import requests
import json
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoConfig
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

@dataclass
class TestCase:
    """Individual test case for recipe generation."""
    name: str
    prompt: str
    expected_elements: List[str]  # Elements that should appear in quality recipes
    edge_case: bool = False
    difficulty: str = "normal"  # normal, challenging, extreme

class CheckpointTester:
    """Comprehensive tester for recipe generation model checkpoints."""
    
    def __init__(self, discord_webhook: str):
        self.discord_webhook = discord_webhook
        self.test_results = []
        self.model = None
        self.tokenizer = None
        self.model_type = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Comprehensive test cases
        self.test_cases = self._create_test_cases()
        
    def _create_test_cases(self) -> List[TestCase]:
        """Create comprehensive test cases covering edge cases and normal scenarios."""
        
        test_cases = []
        
        # NORMAL CASES - Quality Recipe Generation
        test_cases.extend([
            TestCase(
                name="Classic Italian Pasta",
                prompt="Create a traditional Italian carbonara recipe with authentic ingredients and proper technique. Include cooking time and serving size.",
                expected_elements=["pasta", "eggs", "cheese", "pancetta", "pepper", "minutes", "serves"],
                difficulty="normal"
            ),
            TestCase(
                name="Healthy Breakfast Bowl",
                prompt="Design a nutritious breakfast bowl with quinoa, fresh fruits, nuts, and a protein source. Include nutritional benefits and preparation steps.",
                expected_elements=["quinoa", "protein", "fruits", "nuts", "nutrition", "steps", "bowl"],
                difficulty="normal"
            ),
            TestCase(
                name="Asian Fusion Stir-Fry",
                prompt="Develop an Asian fusion stir-fry recipe combining traditional techniques with modern ingredients. Include sauce preparation and cooking order.",
                expected_elements=["stir-fry", "sauce", "vegetables", "protein", "oil", "garlic", "order"],
                difficulty="normal"
            ),
            TestCase(
                name="Gourmet Dessert",
                prompt="Create an elegant chocolate dessert suitable for fine dining with multiple components and plating instructions.",
                expected_elements=["chocolate", "elegant", "components", "plating", "temperature", "texture"],
                difficulty="challenging"
            ),
            TestCase(
                name="Seasonal Soup",
                prompt="Develop a warming autumn soup using seasonal vegetables and herbs with depth of flavor and garnish suggestions.",
                expected_elements=["soup", "vegetables", "herbs", "season", "flavor", "garnish", "simmer"],
                difficulty="normal"
            )
        ])
        
        # EDGE CASES - Complex and Challenging Scenarios
        test_cases.extend([
            TestCase(
                name="Molecular Gastronomy",
                prompt="Create a molecular gastronomy dish using spherification technique with caviar pearls and innovative presentation methods.",
                expected_elements=["spherification", "technique", "pearls", "innovative", "molecular"],
                edge_case=True,
                difficulty="extreme"
            ),
            TestCase(
                name="Extreme Dietary Restrictions",
                prompt="Design a recipe that is simultaneously vegan, gluten-free, keto-friendly, and nut-free while remaining flavorful and satisfying.",
                expected_elements=["vegan", "gluten-free", "keto", "nut-free", "flavorful", "satisfying"],
                edge_case=True,
                difficulty="extreme"
            ),
            TestCase(
                name="Ancient Technique Recreation",
                prompt="Recreate a medieval banquet centerpiece using only ingredients and techniques available in the 14th century.",
                expected_elements=["medieval", "historical", "traditional", "period", "authentic"],
                edge_case=True,
                difficulty="extreme"
            ),
            TestCase(
                name="Emergency Survival Cooking",
                prompt="Create a nutritious meal using only shelf-stable pantry ingredients during a power outage without modern appliances.",
                expected_elements=["pantry", "shelf-stable", "no-cook", "nutrition", "emergency", "simple"],
                edge_case=True,
                difficulty="challenging"
            ),
            TestCase(
                name="Professional Competition Dish",
                prompt="Design a competition-worthy dish for Iron Chef with mystery ingredient sea urchin, requiring three cooking methods and artistic presentation.",
                expected_elements=["competition", "sea urchin", "three methods", "artistic", "professional", "presentation"],
                edge_case=True,
                difficulty="extreme"
            ),
            TestCase(
                name="Fermentation Science",
                prompt="Develop a complex fermentation project involving multiple stages over several months with precise pH and temperature control.",
                expected_elements=["fermentation", "stages", "months", "pH", "temperature", "control", "science"],
                edge_case=True,
                difficulty="extreme"
            ),
            TestCase(
                name="High-Altitude Baking",
                prompt="Adapt a delicate soufflé recipe for baking at 8,000 feet elevation with proper adjustments for air pressure and humidity.",
                expected_elements=["altitude", "soufflé", "elevation", "adjustments", "pressure", "humidity"],
                edge_case=True,
                difficulty="challenging"
            ),
            TestCase(
                name="Therapeutic Cooking",
                prompt="Create a recipe specifically designed for cancer patients undergoing chemotherapy, addressing taste changes and nutritional needs.",
                expected_elements=["therapeutic", "cancer", "chemotherapy", "taste changes", "nutritional", "gentle"],
                edge_case=True,
                difficulty="challenging"
            )
        ])
        
        # CULTURAL FUSION CASES
        test_cases.extend([
            TestCase(
                name="Multi-Cultural Fusion",
                prompt="Blend Japanese, Mexican, and French culinary traditions into a cohesive dish that honors all three cultures authentically.",
                expected_elements=["Japanese", "Mexican", "French", "fusion", "authentic", "traditions"],
                difficulty="challenging"
            ),
            TestCase(
                name="Regional American BBQ",
                prompt="Compare and contrast Texas, Carolina, and Kansas City BBQ styles in a single recipe showcasing regional differences.",
                expected_elements=["Texas", "Carolina", "Kansas City", "BBQ", "regional", "differences", "smoke"],
                difficulty="challenging"
            )
        ])
        
        # TECHNICAL PRECISION CASES
        test_cases.extend([
            TestCase(
                name="Scientific Precision Baking",
                prompt="Create a French macaron recipe with exact gram measurements, humidity requirements, and troubleshooting guide for common failures.",
                expected_elements=["macaron", "grams", "humidity", "precise", "troubleshooting", "technique"],
                difficulty="challenging"
            ),
            TestCase(
                name="Temperature-Critical Cooking",
                prompt="Develop a perfect medium-rare steak recipe using sous vide with reverse sear, including exact temperatures and timing.",
                expected_elements=["sous vide", "medium-rare", "temperature", "reverse sear", "timing", "precise"],
                difficulty="challenging"
            )
        ])
        
        return test_cases
    
    def load_model_from_checkpoint(self, checkpoint_path: str) -> bool:
        """Load model and tokenizer from checkpoint with auto-detection."""
        try:
            print(f"🔄 Loading model from: {checkpoint_path}")
            
            # Load tokenizer first
            self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Auto-detect model type
            config = AutoConfig.from_pretrained(checkpoint_path)
            print(f"📋 Detected model type: {config.model_type}")
            
            # Load model based on type
            if config.model_type in ['t5', 'mt5', 'bart', 'pegasus', 'mbart']:
                # Seq2Seq models
                print(f"🔄 Loading as Seq2Seq model...")
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.model_type = "seq2seq"
            else:
                # Causal LM models (GPT-2, GPT-Neo, etc.)
                print(f"🔄 Loading as Causal Language model...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    checkpoint_path,
                    torch_dtype=torch.bfloat16,
                    device_map="auto" if torch.cuda.is_available() else None
                )
                self.model_type = "causal"
            
            if not torch.cuda.is_available():
                self.model = self.model.to(self.device)
            
            self.model.eval()  # Set to evaluation mode
            
            print(f"✅ Model loaded successfully")
            print(f"   Device: {self.device}")
            print(f"   Model type: {type(self.model).__name__} ({self.model_type})")
            print(f"   Parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            traceback.print_exc()
            return False
    
    def generate_recipe(self, prompt: str, max_length: int = 512) -> Dict[str, Any]:
        """Generate a recipe with detailed metrics for both seq2seq and causal models."""
        if not self.model or not self.tokenizer:
            return {"error": "Model not loaded"}
        
        start_time = time.time()
        
        try:
            if self.model_type == "seq2seq":
                # Seq2Seq generation (T5, BART, etc.)
                formatted_prompt = f"Create a complete recipe with ingredients and instructions: {prompt}"
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.pad_token_id,
                        num_beams=3,
                        early_stopping=True,
                        length_penalty=1.0
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                recipe = generated_text.replace(formatted_prompt, "").strip()
                if not recipe:
                    recipe = generated_text.strip()
                    
            else:
                # Causal LM generation (GPT-2, etc.)
                formatted_prompt = f"Recipe: {prompt}\n\nIngredients:\n"
                inputs = self.tokenizer(
                    formatted_prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=max_length,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9,
                        top_k=50,
                        repetition_penalty=1.2,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id
                    )
                
                generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                # For causal models, the prompt is included in output
                recipe = generated_text[len(formatted_prompt):].strip()
                if not recipe:
                    recipe = generated_text.strip()
            
            generation_time = time.time() - start_time
            
            # Calculate metrics
            word_count = len(recipe.split())
            char_count = len(recipe)
            tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
            
            return {
                "recipe": recipe,
                "generation_time": generation_time,
                "word_count": word_count,
                "char_count": char_count,
                "tokens_generated": max(0, tokens_generated),
                "tokens_per_second": max(0, tokens_generated) / generation_time if generation_time > 0 else 0
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "generation_time": time.time() - start_time
            }
    
    def evaluate_recipe_quality(self, recipe: str, test_case: TestCase) -> Dict[str, Any]:
        """Evaluate the quality of a generated recipe."""
        
        recipe_lower = recipe.lower()
        
        # Check for expected elements
        elements_found = []
        elements_missing = []
        
        for element in test_case.expected_elements:
            if element.lower() in recipe_lower:
                elements_found.append(element)
            else:
                elements_missing.append(element)
        
        coverage_score = len(elements_found) / len(test_case.expected_elements) if test_case.expected_elements else 1.0
        
        # Enhanced quality metrics with more keywords
        has_ingredients = any(word in recipe_lower for word in [
            "ingredients:", "ingredient", "cup", "tablespoon", "teaspoon", "oz", "lb", "gram", 
            "add", "mix", "combine", "salt", "pepper", "oil", "water", "flour", "butter"
        ])
        has_instructions = any(word in recipe_lower for word in [
            "instructions:", "steps:", "directions:", "method:", "1.", "first", "then", "next",
            "heat", "cook", "bake", "stir", "mix", "prepare", "serve", "season", "place"
        ])
        has_timing = any(word in recipe_lower for word in [
            "minutes", "hours", "cook", "bake", "simmer", "boil", "time", "until", "for", "about"
        ])
        has_quantities = any(char.isdigit() for char in recipe)
        
        structure_score = sum([has_ingredients, has_instructions, has_timing, has_quantities]) / 4.0
        
        # Length assessment
        word_count = len(recipe.split())
        length_appropriate = 50 <= word_count <= 800  # Reasonable recipe length
        
        # Overall quality score
        quality_score = (coverage_score * 0.4 + structure_score * 0.4 + (1.0 if length_appropriate else 0.5) * 0.2)
        
        return {
            "coverage_score": coverage_score,
            "structure_score": structure_score,
            "quality_score": quality_score,
            "elements_found": elements_found,
            "elements_missing": elements_missing,
            "has_ingredients": has_ingredients,
            "has_instructions": has_instructions,
            "has_timing": has_timing,
            "has_quantities": has_quantities,
            "length_appropriate": length_appropriate,
            "word_count": word_count
        }
    
    def run_single_test(self, test_case: TestCase) -> Dict[str, Any]:
        """Run a single test case."""
        print(f"🧪 Testing: {test_case.name} ({'EDGE CASE' if test_case.edge_case else 'NORMAL'})")
        
        start_time = time.time()
        
        # Generate recipe
        generation_result = self.generate_recipe(test_case.prompt)
        
        if "error" in generation_result:
            return {
                "test_case": test_case.name,
                "status": "FAILED",
                "error": generation_result["error"],
                "generation_time": generation_result.get("generation_time", 0),
                "total_time": time.time() - start_time
            }
        
        # Evaluate quality
        quality_result = self.evaluate_recipe_quality(generation_result["recipe"], test_case)
        
        total_time = time.time() - start_time
        
        # Determine pass/fail
        min_quality_threshold = 0.3 if test_case.edge_case else 0.5
        status = "PASSED" if quality_result["quality_score"] >= min_quality_threshold else "FAILED"
        
        result = {
            "test_case": test_case.name,
            "status": status,
            "prompt": test_case.prompt,
            "recipe": generation_result["recipe"],
            "edge_case": test_case.edge_case,
            "difficulty": test_case.difficulty,
            "generation_time": generation_result["generation_time"],
            "total_time": total_time,
            "quality_score": quality_result["quality_score"],
            "coverage_score": quality_result["coverage_score"],
            "structure_score": quality_result["structure_score"],
            "word_count": generation_result["word_count"],
            "tokens_per_second": generation_result["tokens_per_second"],
            "elements_found": quality_result["elements_found"],
            "elements_missing": quality_result["elements_missing"],
            "has_structure": all([
                quality_result["has_ingredients"],
                quality_result["has_instructions"],
                quality_result["has_timing"]
            ])
        }
        
        return result
    
    def run_comprehensive_test(self, checkpoint_path: str, parallel_tests: int = 3) -> Dict[str, Any]:
        """Run comprehensive test suite on a checkpoint."""
        
        print(f"\n🚀 COMPREHENSIVE CHECKPOINT TESTING")
        print(f"📁 Checkpoint: {checkpoint_path}")
        print(f"🧪 Total test cases: {len(self.test_cases)}")
        print(f"⚡ Parallel tests: {parallel_tests}")
        print("=" * 80)
        
        # Load model
        if not self.load_model_from_checkpoint(checkpoint_path):
            return {"error": "Failed to load model"}
        
        # Run tests in parallel for speed
        start_time = time.time()
        results = []
        
        # Split tests into normal and edge cases for different threading
        normal_cases = [tc for tc in self.test_cases if not tc.edge_case]
        edge_cases = [tc for tc in self.test_cases if tc.edge_case]
        
        print(f"🧪 Running {len(normal_cases)} normal tests...")
        with ThreadPoolExecutor(max_workers=parallel_tests) as executor:
            future_to_test = {executor.submit(self.run_single_test, tc): tc for tc in normal_cases}
            
            for future in as_completed(future_to_test):
                result = future.result()
                results.append(result)
                status_emoji = "✅" if result["status"] == "PASSED" else "❌"
                print(f"  {status_emoji} {result['test_case']}: {result['quality_score']:.3f} quality")
        
        print(f"🧪 Running {len(edge_cases)} edge case tests...")
        with ThreadPoolExecutor(max_workers=max(1, parallel_tests // 2)) as executor:  # Fewer parallel for edge cases
            future_to_test = {executor.submit(self.run_single_test, tc): tc for tc in edge_cases}
            
            for future in as_completed(future_to_test):
                result = future.result()
                results.append(result)
                status_emoji = "✅" if result["status"] == "PASSED" else "❌"
                print(f"  {status_emoji} {result['test_case']}: {result['quality_score']:.3f} quality")
        
        total_time = time.time() - start_time
        
        # Calculate aggregate metrics
        passed_tests = [r for r in results if r["status"] == "PASSED"]
        failed_tests = [r for r in results if r["status"] == "FAILED"]
        normal_results = [r for r in results if not r.get("edge_case", False)]
        edge_results = [r for r in results if r.get("edge_case", False)]
        
        aggregate_metrics = {
            "checkpoint_path": checkpoint_path,
            "total_tests": len(results),
            "passed": len(passed_tests),
            "failed": len(failed_tests),
            "pass_rate": len(passed_tests) / len(results) if results else 0,
            "normal_pass_rate": len([r for r in normal_results if r["status"] == "PASSED"]) / len(normal_results) if normal_results else 0,
            "edge_case_pass_rate": len([r for r in edge_results if r["status"] == "PASSED"]) / len(edge_results) if edge_results else 0,
            "avg_quality_score": sum(r["quality_score"] for r in results) / len(results) if results else 0,
            "avg_generation_time": sum(r["generation_time"] for r in results) / len(results) if results else 0,
            "avg_tokens_per_second": sum(r["tokens_per_second"] for r in results if r.get("tokens_per_second")) / len([r for r in results if r.get("tokens_per_second")]) if results else 0,
            "total_test_time": total_time,
            "individual_results": results
        }
        
        return aggregate_metrics
    
    def send_discord_notification(self, results: Dict[str, Any]):
        """Send comprehensive test results to Discord."""
        
        try:
            # Determine overall status
            pass_rate = results["pass_rate"]
            if pass_rate >= 0.8:
                color = 0x00ff00  # Green - excellent
                status_emoji = "🎉"
                status_text = "EXCELLENT"
            elif pass_rate >= 0.6:
                color = 0xffaa00  # Orange - good
                status_emoji = "✅"
                status_text = "GOOD"
            else:
                color = 0xff0000  # Red - needs improvement
                status_emoji = "⚠️"
                status_text = "NEEDS IMPROVEMENT"
            
            # Create embed with special handling for final model
            is_final = results.get("is_final_model", False)
            title_prefix = "🏆 FINAL MODEL - " if is_final else ""
            
            embed = {
                "title": f"{title_prefix}{status_emoji} Recipe Model Checkpoint Test Results",
                "description": f"**Status: {status_text}** (Pass Rate: {pass_rate:.1%})" + (" 🏆 **FINAL TRAINED MODEL**" if is_final else ""),
                "color": color,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                "fields": [
                    {
                        "name": "📊 Overall Results",
                        "value": f"✅ Passed: {results['passed']}/{results['total_tests']}\n❌ Failed: {results['failed']}/{results['total_tests']}\n📈 Pass Rate: {pass_rate:.1%}",
                        "inline": True
                    },
                    {
                        "name": "🧪 Test Categories",
                        "value": f"🟢 Normal: {results['normal_pass_rate']:.1%}\n🔥 Edge Cases: {results['edge_case_pass_rate']:.1%}",
                        "inline": True
                    },
                    {
                        "name": "⚡ Performance",
                        "value": f"🏃 Avg Gen Time: {results['avg_generation_time']:.2f}s\n🚀 Tokens/sec: {results['avg_tokens_per_second']:.1f}\n⏱️ Total Time: {results['total_test_time']:.1f}s",
                        "inline": True
                    },
                    {
                        "name": "🎯 Quality Metrics",
                        "value": f"📊 Avg Quality: {results['avg_quality_score']:.3f}\n📝 Recipe Structure: {'✅' if results['avg_quality_score'] > 0.5 else '⚠️'}",
                        "inline": True
                    },
                    {
                        "name": "📁 Checkpoint Tested",
                        "value": f"`{os.path.basename(results['checkpoint_path'])}`",
                        "inline": False
                    }
                ]
            }
            
            # Add sample results
            best_results = sorted(results["individual_results"], key=lambda x: x["quality_score"], reverse=True)[:3]
            worst_results = sorted(results["individual_results"], key=lambda x: x["quality_score"])[:2]
            
            if best_results:
                best_text = "\n".join([f"✅ {r['test_case']}: {r['quality_score']:.3f}" for r in best_results])
                embed["fields"].append({
                    "name": "🏆 Top Performing Tests",
                    "value": best_text,
                    "inline": True
                })
            
            if worst_results and any(r["status"] == "FAILED" for r in worst_results):
                worst_text = "\n".join([f"❌ {r['test_case']}: {r['quality_score']:.3f}" for r in worst_results if r["status"] == "FAILED"])
                if worst_text:
                    embed["fields"].append({
                        "name": "⚠️ Failed Tests",
                        "value": worst_text,
                        "inline": True
                    })
            
            # Add edge case highlights
            edge_passed = [r for r in results["individual_results"] if r.get("edge_case") and r["status"] == "PASSED"]
            if edge_passed:
                edge_text = "\n".join([f"🔥 {r['test_case']}" for r in edge_passed[:3]])
                embed["fields"].append({
                    "name": "🔥 Edge Cases Passed",
                    "value": edge_text,
                    "inline": False
                })
            
            payload = {
                "embeds": [embed],
                "username": "Chef Genius Checkpoint Tester"
            }
            
            response = requests.post(self.discord_webhook, json=payload, timeout=10)
            response.raise_for_status()
            
            print(f"\n🔔 Discord notification sent successfully!")
            
        except Exception as e:
            print(f"⚠️ Failed to send Discord notification: {e}")
    
    def find_checkpoints(self, models_dir: str = "models") -> List[str]:
        """Find all available checkpoints."""
        checkpoints = []
        
        models_path = Path(models_dir)
        if not models_path.exists():
            print(f"⚠️ Models directory not found: {models_dir}")
            return checkpoints
        
        # Look for checkpoint directories
        for item in models_path.rglob("checkpoint-*"):
            if item.is_dir():
                # Verify it's a valid checkpoint
                if (item / "config.json").exists() and (item / "model.safetensors").exists():
                    checkpoints.append(str(item))
        
        # Also look for main model directories
        for item in models_path.iterdir():
            if item.is_dir() and not item.name.startswith("checkpoint-"):
                if (item / "config.json").exists() and (item / "model.safetensors").exists():
                    checkpoints.append(str(item))
        
        return sorted(checkpoints)
    
    def test_all_checkpoints(self, models_dir: str = "models"):
        """Test all available checkpoints."""
        checkpoints = self.find_checkpoints(models_dir)
        
        if not checkpoints:
            print(f"❌ No checkpoints found in {models_dir}")
            return
        
        print(f"🔍 Found {len(checkpoints)} checkpoints to test:")
        for cp in checkpoints:
            print(f"  📁 {cp}")
        print()
        
        for i, checkpoint in enumerate(checkpoints, 1):
            print(f"\n🧪 Testing checkpoint {i}/{len(checkpoints)}: {os.path.basename(checkpoint)}")
            print("=" * 80)
            
            try:
                results = self.run_comprehensive_test(checkpoint)
                
                if "error" not in results:
                    print(f"\n📊 RESULTS SUMMARY:")
                    print(f"   Pass Rate: {results['pass_rate']:.1%}")
                    print(f"   Quality Score: {results['avg_quality_score']:.3f}")
                    print(f"   Performance: {results['avg_tokens_per_second']:.1f} tokens/sec")
                    
                    # Send to Discord
                    self.send_discord_notification(results)
                else:
                    print(f"❌ Test failed: {results['error']}")
                    
            except Exception as e:
                print(f"💥 Checkpoint test crashed: {e}")
                traceback.print_exc()
            
            # Clear GPU memory between checkpoints
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def main():
    """Main function for running checkpoint tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Checkpoint Tester for Recipe Generation Model')
    parser.add_argument('--checkpoint', type=str, help='Specific checkpoint path to test')
    parser.add_argument('--final-model', action='store_true', help='Test only the final/latest trained model')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory containing model checkpoints')
    parser.add_argument('--discord-webhook', type=str, 
                       default='https://discord.com/api/webhooks/1386109570283343953/uGkhj9dpuCg09SbKzZ0Tx2evugJrchQv-nrq3w0r_xi3w8si-XBpQJuxq_p_bcQlhB9W',
                       help='Discord webhook URL for notifications')
    parser.add_argument('--parallel', type=int, default=3, help='Number of parallel tests to run')
    
    args = parser.parse_args()
    
    print("🍳 ENTERPRISE RECIPE GENERATION MODEL CHECKPOINT TESTER")
    print("🧪 Comprehensive testing with edge cases and quality assessment")
    print("=" * 80)
    
    # Create tester
    tester = CheckpointTester(discord_webhook=args.discord_webhook)
    
    if args.checkpoint:
        # Test specific checkpoint
        if not os.path.exists(args.checkpoint):
            print(f"❌ Checkpoint not found: {args.checkpoint}")
            return
        
        print(f"🎯 Testing specific checkpoint: {args.checkpoint}")
        results = tester.run_comprehensive_test(args.checkpoint, parallel_tests=args.parallel)
        
        if "error" not in results:
            tester.send_discord_notification(results)
        else:
            print(f"❌ Test failed: {results['error']}")
    elif args.final_model:
        # Test only the final/latest model
        checkpoints = tester.find_checkpoints(args.models_dir)
        if not checkpoints:
            print(f"❌ No checkpoints found in {args.models_dir}")
            return
        
        # Find the final model - either highest numbered checkpoint or main model
        final_checkpoint = None
        
        # Look for the main model directory first (non-checkpoint)
        main_models = [cp for cp in checkpoints if not os.path.basename(cp).startswith("checkpoint-")]
        if main_models:
            final_checkpoint = main_models[0]  # Use the main model
        else:
            # Find highest numbered checkpoint
            numbered_checkpoints = []
            for cp in checkpoints:
                basename = os.path.basename(cp)
                if basename.startswith("checkpoint-"):
                    try:
                        num = int(basename.split("-")[1])
                        numbered_checkpoints.append((num, cp))
                    except (IndexError, ValueError):
                        continue
            
            if numbered_checkpoints:
                # Get the highest numbered checkpoint
                numbered_checkpoints.sort(key=lambda x: x[0], reverse=True)
                final_checkpoint = numbered_checkpoints[0][1]
        
        if final_checkpoint:
            print(f"🏆 Testing FINAL MODEL: {os.path.basename(final_checkpoint)}")
            print(f"📁 Path: {final_checkpoint}")
            print("=" * 80)
            
            results = tester.run_comprehensive_test(final_checkpoint, parallel_tests=args.parallel)
            
            if "error" not in results:
                # Add special marking for final model in Discord
                results["is_final_model"] = True
                tester.send_discord_notification(results)
                
                # Print summary
                print(f"\n🎉 FINAL MODEL TEST COMPLETE!")
                print(f"   📊 Pass Rate: {results['pass_rate']:.1%}")
                print(f"   🎯 Quality Score: {results['avg_quality_score']:.3f}")
                print(f"   ⚡ Performance: {results['avg_tokens_per_second']:.1f} tokens/sec")
            else:
                print(f"❌ Final model test failed: {results['error']}")
        else:
            print(f"❌ Could not find final model in {args.models_dir}")
    else:
        # Test all checkpoints
        tester.test_all_checkpoints(args.models_dir)

if __name__ == "__main__":
    main()