#!/usr/bin/env python3
"""
Improved Pork Belly Recipe Generator
Uses the new BaseRecipeGenerator with all enhancements
"""

import os
import sys
import argparse
from typing import List
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from base_recipe_generator import BaseRecipeGenerator, RecipePrompt, GenerationConfig
from recipe_metrics import RecipeMetrics
from recipe_exporters import RecipeExporter
from discord_notifier import DiscordNotifier, ProgressTracker
from test_recipe_generator import run_diversity_check


class ImprovedPorkBellyGenerator(BaseRecipeGenerator):
    """Improved pork belly recipe generator with enhanced features."""

    def get_recipe_type(self) -> str:
        """Return the recipe type."""
        return "Pork Belly"

    def _create_prompts(self) -> List[RecipePrompt]:
        """Create pork belly recipe prompts."""
        prompts = []

        # TRADITIONAL TECHNIQUES (5 prompts)
        prompts.extend([
            RecipePrompt(
                name="Classic Crispy Pork Belly",
                prompt="Create a traditional crispy pork belly recipe with perfectly rendered fat and crackling skin. Include proper scoring techniques and oven temperature control.",
                category="traditional",
                difficulty="normal",
                expected_features=["scoring", "skin", "crackling", "rendering", "temperature", "crispy"]
            ),
            RecipePrompt(
                name="Chinese Red-Braised Pork Belly (Hong Shao Rou)",
                prompt="Design an authentic Chinese red-braised pork belly recipe with soy sauce, rock sugar, and Shaoxing wine. Include traditional technique for achieving tender, glossy meat.",
                category="traditional",
                difficulty="normal",
                expected_features=["soy sauce", "rock sugar", "Shaoxing wine", "braising", "tender", "glossy"]
            ),
            RecipePrompt(
                name="Korean-Style Pan-Seared Pork Belly",
                prompt="Create a Korean-style pan-seared pork belly recipe with proper technique for achieving crispy exterior in a skillet, including traditional banchan suggestions and wrapping techniques.",
                category="traditional",
                difficulty="normal",
                expected_features=["pan-seared", "banchan", "lettuce wraps", "Korean", "skillet", "crispy"]
            ),
            RecipePrompt(
                name="Italian Porchetta-Style Pork Belly",
                prompt="Develop an Italian porchetta-inspired pork belly recipe with herb stuffing, proper rolling technique, and slow roasting for crispy exterior and tender interior.",
                category="traditional",
                difficulty="challenging",
                expected_features=["herbs", "rolling", "stuffing", "slow roasting", "crispy", "tender"]
            ),
            RecipePrompt(
                name="Southern-Style Oven-Braised Pork Belly",
                prompt="Create a Southern-style oven-braised pork belly recipe with proper cubing, seasoning, and slow oven cooking for caramelized, tender bites.",
                category="traditional",
                difficulty="challenging",
                expected_features=["cubing", "oven-braised", "caramelized", "Southern", "seasoning", "tender"]
            )
        ])

        # ROASTING & PAN COOKING (4 prompts)
        prompts.extend([
            RecipePrompt(
                name="Low and Slow Oven-Roasted Pork Belly",
                prompt="Design a low and slow oven-roasted pork belly recipe with temperature control, proper scoring, and timing for crispy skin and tender meat.",
                category="roasting",
                difficulty="challenging",
                expected_features=["oven-roasted", "scoring", "crispy skin", "temperature", "low and slow", "tender"]
            ),
            RecipePrompt(
                name="Pan-Seared and Oven-Finished Pork Belly Bites",
                prompt="Create pan-seared pork belly bites finished in the oven with proper cubing, searing technique, and oven glazing for caramelized results.",
                category="roasting",
                difficulty="challenging",
                expected_features=["pan-seared", "cubing", "oven-finished", "glazing", "caramelized", "bites"]
            ),
            RecipePrompt(
                name="Oven-Cured and Roasted Pork Belly",
                prompt="Develop an oven-cured and roasted pork belly recipe for homemade bacon-style results, including dry curing process and slow roasting techniques.",
                category="roasting",
                difficulty="extreme",
                expected_features=["oven-cured", "dry curing", "bacon-style", "homemade", "slow roasting", "process"]
            ),
            RecipePrompt(
                name="Herb-Crusted Oven-Roasted Pork Belly",
                prompt="Create an herb-crusted oven-roasted pork belly recipe with complementary herb blend, proper coating technique, and perfect roasting method.",
                category="roasting",
                difficulty="normal",
                expected_features=["herb-crusted", "oven-roasted", "herb blend", "coating", "roasting method"]
            )
        ])

        # BRAISING & SLOW COOKING (4 prompts)
        prompts.extend([
            RecipePrompt(
                name="Beer-Braised Pork Belly",
                prompt="Design a beer-braised pork belly recipe with proper beer selection, vegetable aromatics, and braising liquid reduction for rich, tender results.",
                category="braising",
                difficulty="normal",
                expected_features=["beer", "braising", "aromatics", "reduction", "tender", "rich"]
            ),
            RecipePrompt(
                name="Miso-Braised Pork Belly Ramen",
                prompt="Create a Japanese-inspired miso-braised pork belly for ramen with proper chashu technique, marinating, and slicing for perfect ramen topping.",
                category="braising",
                difficulty="challenging",
                expected_features=["miso", "chashu", "ramen", "marinating", "Japanese", "slicing"]
            ),
            RecipePrompt(
                name="Wine-Braised Pork Belly Confit",
                prompt="Develop a French-style wine-braised pork belly confit with proper wine selection, herb bouquet, and low-temperature cooking for ultimate tenderness.",
                category="braising",
                difficulty="challenging",
                expected_features=["wine", "confit", "French", "herb bouquet", "low temperature", "tenderness"]
            ),
            RecipePrompt(
                name="Coconut Curry Braised Pork Belly",
                prompt="Create a Southeast Asian coconut curry braised pork belly with proper spice balance, coconut milk technique, and aromatic garnishes.",
                category="braising",
                difficulty="normal",
                expected_features=["coconut curry", "Southeast Asian", "spice balance", "coconut milk", "aromatic", "garnishes"]
            )
        ])

        # FUSION CUISINE (4 prompts)
        prompts.extend([
            RecipePrompt(
                name="Korean-Mexican Pork Belly Tacos",
                prompt="Design fusion pork belly tacos combining Korean gochujang flavors with Mexican tortilla techniques, including kimchi slaw and sesame-lime crema.",
                category="fusion",
                difficulty="challenging",
                expected_features=["Korean", "Mexican", "gochujang", "tacos", "kimchi", "sesame-lime"]
            ),
            RecipePrompt(
                name="Vietnamese-Italian Pork Belly Banh Mi Pizza",
                prompt="Create an innovative pizza combining Vietnamese banh mi flavors with Italian pizza technique, featuring pork belly, pickled vegetables, and cilantro.",
                category="fusion",
                difficulty="extreme",
                expected_features=["Vietnamese", "Italian", "banh mi", "pizza", "pickled vegetables", "cilantro"]
            ),
            RecipePrompt(
                name="Thai-French Pork Belly Cassoulet",
                prompt="Develop a fusion cassoulet using Thai flavors and pork belly, incorporating lemongrass, fish sauce, and traditional French bean cooking techniques.",
                category="fusion",
                difficulty="extreme",
                expected_features=["Thai", "French", "cassoulet", "lemongrass", "fish sauce", "beans"]
            ),
            RecipePrompt(
                name="Indian-Southern Pork Belly Curry",
                prompt="Create a spiced pork belly curry combining Indian spice techniques with Southern US comfort food elements, including proper spice tempering and slow cooking.",
                category="fusion",
                difficulty="challenging",
                expected_features=["Indian spices", "Southern", "curry", "tempering", "slow cooking", "comfort food"]
            )
        ])

        # CREATIVE TECHNIQUES (3 prompts)
        prompts.extend([
            RecipePrompt(
                name="Sous Vide Pork Belly with Torched Finish",
                prompt="Design a modern sous vide pork belly recipe with precise time and temperature, followed by torching technique for perfect texture contrast.",
                category="creative",
                difficulty="extreme",
                expected_features=["sous vide", "precise temperature", "torching", "texture contrast", "modern", "technique"]
            ),
            RecipePrompt(
                name="Pork Belly Bao Buns from Scratch",
                prompt="Create homemade bao buns with perfectly steamed pork belly filling, including dough preparation, steaming technique, and traditional garnishes.",
                category="creative",
                difficulty="extreme",
                expected_features=["bao buns", "steamed", "dough", "steaming", "homemade", "garnishes"]
            ),
            RecipePrompt(
                name="Deconstructed Pork Belly Ramen Bowl",
                prompt="Design a modern deconstructed ramen presentation featuring pork belly as the centerpiece with innovative plating and molecular gastronomy elements.",
                category="creative",
                difficulty="extreme",
                expected_features=["deconstructed", "modern", "centerpiece", "plating", "molecular gastronomy", "innovative"]
            )
        ])

        return prompts

    def analyze_recipe_quality(self, recipe: str, prompt: RecipePrompt) -> dict:
        """Enhanced quality analysis for pork belly recipes."""
        # Get base analysis
        base_analysis = super().analyze_recipe_quality(recipe, prompt)

        # Add pork belly specific checks
        recipe_lower = recipe.lower()

        has_pork_belly = any(term in recipe_lower for term in ["pork belly", "pork-belly", "belly"])
        has_cooking_technique = any(tech in recipe_lower for tech in [
            "render", "score", "sear", "braise", "roast", "crispy", "tender"
        ])

        # Pork belly specific quality adjustments
        if has_pork_belly:
            base_analysis["overall_quality"] *= 1.1  # Boost if pork belly is mentioned
        else:
            base_analysis["overall_quality"] *= 0.7  # Penalize if pork belly not mentioned

        if has_cooking_technique:
            base_analysis["overall_quality"] *= 1.05

        # Cap at 1.0
        base_analysis["overall_quality"] = min(1.0, base_analysis["overall_quality"])

        base_analysis["has_pork_belly"] = has_pork_belly
        base_analysis["has_cooking_technique"] = has_cooking_technique

        return base_analysis


def main():
    """Main function for improved pork belly generation."""
    parser = argparse.ArgumentParser(description='Improved Pork Belly Recipe Generator')
    parser.add_argument('--checkpoint', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--discord-webhook', type=str, help='Discord webhook URL')
    parser.add_argument('--parallel', type=int, default=2, help='Parallel generations')
    parser.add_argument('--output', type=str, help='Output base filename')
    parser.add_argument('--enable-beam-search', action='store_true', help='Enable beam search')
    parser.add_argument('--quality-threshold', type=float, default=0.6, help='Quality threshold')
    parser.add_argument('--max-retries', type=int, default=2, help='Max retries per recipe')
    parser.add_argument('--export-formats', nargs='+',
                       choices=['markdown', 'html', 'pdf', 'text', 'json', 'all'],
                       default=['markdown', 'json'],
                       help='Export formats')
    parser.add_argument('--prompts-file', type=str, help='Load prompts from YAML/JSON file')
    parser.add_argument('--save-prompts', type=str, help='Save prompts to YAML/JSON file')
    parser.add_argument('--check-diversity', action='store_true',
                       help='Run diversity check on results')

    args = parser.parse_args()

    print("🥓 IMPROVED PORK BELLY RECIPE GENERATOR")
    print("=" * 80)

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        return 1

    # Create generation config
    config = GenerationConfig(
        generation_modes=["greedy", "normal", "creative"],
        enable_beam_search=args.enable_beam_search,
        retry_on_failure=True,
        max_retries=args.max_retries,
        quality_threshold=args.quality_threshold,
        enable_few_shot=True
    )

    # Create generator
    generator = ImprovedPorkBellyGenerator(
        discord_webhook=args.discord_webhook,
        config=config
    )

    # Load or save prompts if specified
    if args.prompts_file:
        print(f"📂 Loading prompts from: {args.prompts_file}")
        generator.recipe_prompts = generator.load_prompts_from_file(args.prompts_file)

    if args.save_prompts:
        print(f"💾 Saving prompts to: {args.save_prompts}")
        generator.save_prompts_to_file(args.save_prompts)

    # Set up Discord notifications
    notifier = None
    if args.discord_webhook:
        notifier = DiscordNotifier(args.discord_webhook, "Pork Belly Generator")
        notifier.send_session_start(
            recipe_type="Pork Belly",
            total_recipes=len(generator.recipe_prompts),
            checkpoint=os.path.basename(args.checkpoint),
            config=config.__dict__
        )

    # Generate recipes
    print(f"\n🚀 Starting generation with {len(generator.recipe_prompts)} recipes...")
    results = generator.generate_all_recipes(args.checkpoint, parallel_generations=args.parallel)

    if "error" in results:
        print(f"❌ Generation failed: {results['error']}")
        if notifier:
            notifier.send_error(results['error'], "Recipe generation failed")
        return 1

    # Print summary
    print(f"\n🎉 GENERATION COMPLETE!")
    print(f"   📊 Success Rate: {results['success_rate']:.1%}")
    print(f"   🎯 Average Quality: {results['avg_quality_score']:.3f}")
    print(f"   ⚡ Performance: {results['avg_tokens_per_second']:.1f} tokens/sec")

    # Run diversity check
    if args.check_diversity:
        diversity_analysis = run_diversity_check(results['individual_results'])

        if notifier and diversity_analysis:
            notifier.send_message(embed={
                "title": "📊 Diversity Analysis",
                "description": diversity_analysis['diagnosis'],
                "color": 0x3498db,
                "fields": [
                    {
                        "name": "Metrics",
                        "value": f"Avg Similarity: {diversity_analysis['avg_similarity']:.3f}\nUnique Ingredients: {diversity_analysis['unique_ingredients']}",
                        "inline": False
                    }
                ]
            })

    # Export results
    base_filename = args.output or f"pork_belly_results_{int(time.time())}"

    if 'all' in args.export_formats:
        RecipeExporter.export_all_formats(results, base_filename)
    else:
        for fmt in args.export_formats:
            if fmt == 'markdown':
                RecipeExporter.export_to_markdown(results, f"{base_filename}.md")
            elif fmt == 'html':
                RecipeExporter.export_to_html(results, f"{base_filename}.html")
            elif fmt == 'pdf':
                RecipeExporter.export_to_pdf(results, f"{base_filename}.pdf")
            elif fmt == 'text':
                RecipeExporter.export_to_text(results, f"{base_filename}.txt")
            elif fmt == 'json':
                RecipeExporter.export_to_json(results, f"{base_filename}.json")

    # Send Discord completion notification
    if notifier:
        generator.send_discord_notification(results)

    # Cleanup
    generator.cleanup()

    print(f"\n✅ All done! Results exported to: {base_filename}.*")

    return 0


if __name__ == "__main__":
    import time
    sys.exit(main())
