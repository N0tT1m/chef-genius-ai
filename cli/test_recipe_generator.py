#!/usr/bin/env python3
"""
Unit Tests and Regression Testing for Recipe Generators
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from recipe_metrics import RecipeMetrics
from base_recipe_generator import RecipePrompt, ValidationResult


class TestRecipeMetrics(unittest.TestCase):
    """Test recipe metrics calculations."""

    def test_bleu_identical_texts(self):
        """Test BLEU score with identical texts."""
        text = "This is a test recipe with ingredients and instructions"
        score = RecipeMetrics.calculate_bleu(text, text)
        self.assertAlmostEqual(score, 1.0, places=2)

    def test_bleu_different_texts(self):
        """Test BLEU score with completely different texts."""
        text1 = "This is a test recipe"
        text2 = "Completely different content here"
        score = RecipeMetrics.calculate_bleu(text1, text2)
        self.assertLess(score, 0.3)

    def test_rouge_identical_texts(self):
        """Test ROUGE score with identical texts."""
        text = "This is a test recipe"
        scores = RecipeMetrics.calculate_rouge(text, text, 'rouge-1')
        self.assertAlmostEqual(scores['f1'], 1.0, places=2)

    def test_rouge_different_texts(self):
        """Test ROUGE score with different texts."""
        text1 = "This is a test"
        text2 = "Completely different"
        scores = RecipeMetrics.calculate_rouge(text1, text2, 'rouge-1')
        self.assertLess(scores['f1'], 0.5)

    def test_coherence_complete_recipe(self):
        """Test coherence with a complete recipe."""
        recipe = """
Classic Pasta Recipe

INGREDIENTS:
- 1 lb pasta
- 2 cloves garlic
- 1/4 cup olive oil
- Salt and pepper

INSTRUCTIONS:
1. Boil water and cook pasta according to package directions
2. Heat olive oil in pan and sauté garlic until fragrant
3. Drain pasta and toss with garlic oil
4. Season with salt and pepper to taste

NOTES:
- Use good quality olive oil for best results
"""
        result = RecipeMetrics.check_coherence(recipe)
        self.assertGreater(result['overall_coherence'], 0.5)
        self.assertTrue(result['has_numbered_steps'])
        self.assertGreater(result['ingredient_coverage'], 0.5)

    def test_coherence_missing_ingredients(self):
        """Test coherence when ingredients aren't used."""
        recipe = """
Test Recipe

INGREDIENTS:
- unused ingredient one
- unused ingredient two
- garlic

INSTRUCTIONS:
1. Just add some random stuff
2. Cook it
"""
        result = RecipeMetrics.check_coherence(recipe)
        self.assertGreater(len(result['unmentioned_ingredients']), 0)

    def test_lexical_diversity(self):
        """Test lexical diversity calculation."""
        # Diverse text
        diverse_text = "This is a test with many different unique words here"
        diversity1 = RecipeMetrics.calculate_lexical_diversity(diverse_text)

        # Repetitive text
        repetitive_text = "test test test test test test test test"
        diversity2 = RecipeMetrics.calculate_lexical_diversity(repetitive_text)

        self.assertGreater(diversity1, diversity2)

    def test_readability(self):
        """Test readability metrics."""
        text = "This is a simple sentence. Here is another one. And a third."
        result = RecipeMetrics.calculate_readability(text)

        self.assertIn('flesch_reading_ease', result)
        self.assertIn('interpretation', result)
        self.assertGreater(result['flesch_reading_ease'], 0)
        self.assertLess(result['flesch_reading_ease'], 100)

    def test_completeness_full_recipe(self):
        """Test completeness with a full recipe."""
        recipe = """
Perfect Grilled Chicken

Serves 4 | Prep: 15 minutes | Cook: 20 minutes

INGREDIENTS:
- 4 chicken breasts
- 2 tbsp olive oil
- Salt and pepper

INSTRUCTIONS:
1. Preheat grill to 400°F
2. Season chicken with salt and pepper
3. Grill for 10 minutes per side

NOTES:
- Use a meat thermometer
"""
        result = RecipeMetrics.analyze_recipe_completeness(recipe)

        self.assertTrue(result['has_ingredients'])
        self.assertTrue(result['has_instructions'])
        self.assertTrue(result['has_servings'])
        self.assertTrue(result['has_time'])
        self.assertTrue(result['has_temperature'])
        self.assertGreater(result['overall_completeness'], 0.7)

    def test_completeness_minimal_recipe(self):
        """Test completeness with minimal recipe."""
        recipe = "Just some random text without proper structure"
        result = RecipeMetrics.analyze_recipe_completeness(recipe)

        self.assertLess(result['overall_completeness'], 0.5)

    def test_comprehensive_evaluation(self):
        """Test comprehensive evaluation."""
        recipe = """
Classic Chocolate Chip Cookies

INGREDIENTS:
- 2 cups flour
- 1 cup butter
- 2 eggs
- 2 cups chocolate chips

INSTRUCTIONS:
1. Preheat oven to 375°F
2. Mix flour and butter in large bowl
3. Beat in eggs until well combined
4. Fold in chocolate chips
5. Drop spoonfuls onto baking sheet
6. Bake for 10 minutes

NOTES:
- Don't overbake for chewy cookies
"""
        result = RecipeMetrics.comprehensive_evaluation(recipe)

        self.assertIn('coherence', result)
        self.assertIn('completeness', result)
        self.assertIn('lexical_diversity', result)
        self.assertIn('readability', result)
        self.assertIn('overall_score', result)
        self.assertGreater(result['overall_score'], 0)


class TestRecipeValidation(unittest.TestCase):
    """Test recipe validation logic."""

    def test_extract_ingredient_names(self):
        """Test ingredient name extraction."""
        ingredients_text = """
- 2 cups all-purpose flour
- 1/2 cup butter, softened
- 2 large eggs
- 1 tsp vanilla extract
"""
        names = RecipeMetrics._extract_ingredient_names(ingredients_text)

        self.assertGreater(len(names), 0)
        # Should extract main ingredients like flour, butter, eggs
        found_flour = any('flour' in name for name in names)
        self.assertTrue(found_flour)

    def test_validation_time_patterns(self):
        """Test time pattern extraction from recipe."""
        import re

        recipe = "Bake for 45 minutes at 350°F. Let rest for 10 minutes."
        time_pattern = r'(\d+)\s*(hour|hr|minute|min)'
        times = re.findall(time_pattern, recipe.lower())

        self.assertEqual(len(times), 2)
        self.assertEqual(times[0], ('45', 'minute'))
        self.assertEqual(times[1], ('10', 'minute'))

    def test_validation_temperature_patterns(self):
        """Test temperature pattern extraction."""
        import re

        recipe = "Preheat oven to 375°F. Grill at 400 degrees."
        temp_pattern = r'(\d+)\s*°?[FCfc]'
        temps = re.findall(temp_pattern, recipe.lower())

        self.assertGreater(len(temps), 0)


class TestRecipeDiversity(unittest.TestCase):
    """Test recipe diversity detection."""

    def test_recipe_similarity(self):
        """Test detection of similar recipes."""
        recipe1 = """
INGREDIENTS:
- pasta
- tomatoes
- garlic

INSTRUCTIONS:
1. Boil pasta
2. Cook tomatoes with garlic
3. Mix together
"""

        recipe2 = """
INGREDIENTS:
- pasta
- tomatoes
- garlic

INSTRUCTIONS:
1. Boil pasta
2. Cook tomatoes with garlic
3. Mix together
"""

        # These are identical, should have high similarity
        bleu = RecipeMetrics.calculate_bleu(recipe1, recipe2)
        self.assertGreater(bleu, 0.9)

    def test_recipe_diversity_different(self):
        """Test that different recipes are detected as different."""
        recipe1 = """
INGREDIENTS:
- chicken
- lemon
- herbs

INSTRUCTIONS:
1. Season chicken with herbs
2. Grill until cooked
3. Squeeze lemon on top
"""

        recipe2 = """
INGREDIENTS:
- pasta
- tomatoes
- garlic

INSTRUCTIONS:
1. Boil pasta
2. Make tomato sauce with garlic
3. Combine and serve
"""

        # These are different, should have low similarity
        bleu = RecipeMetrics.calculate_bleu(recipe1, recipe2)
        self.assertLess(bleu, 0.3)


class TestDiversityAnalyzer:
    """Analyzer to detect if all recipes are the same."""

    @staticmethod
    def analyze_recipe_diversity(recipes: list) -> dict:
        """
        Analyze diversity across multiple recipes.

        Args:
            recipes: List of recipe texts

        Returns:
            Dictionary with diversity metrics
        """
        if len(recipes) < 2:
            return {"error": "Need at least 2 recipes to analyze diversity"}

        # Calculate pairwise BLEU scores
        similarities = []
        for i in range(len(recipes)):
            for j in range(i + 1, len(recipes)):
                bleu = RecipeMetrics.calculate_bleu(recipes[i], recipes[j])
                similarities.append(bleu)

        avg_similarity = sum(similarities) / len(similarities) if similarities else 0

        # Calculate unique ingredients across all recipes
        all_ingredients = set()
        for recipe in recipes:
            coherence = RecipeMetrics.check_coherence(recipe)
            for ing in coherence['mentioned_ingredients']:
                all_ingredients.add(ing.lower())

        # Calculate lexical diversity for each recipe
        diversities = [RecipeMetrics.calculate_lexical_diversity(r) for r in recipes]
        avg_diversity = sum(diversities) / len(diversities)

        # Check if recipes are too similar (possible early checkpoint issue)
        is_diverse = avg_similarity < 0.7  # Threshold for diversity
        has_variety = len(all_ingredients) > len(recipes) * 0.5  # At least 50% unique ingredient names

        return {
            "avg_similarity": avg_similarity,
            "avg_lexical_diversity": avg_diversity,
            "unique_ingredients": len(all_ingredients),
            "total_recipes": len(recipes),
            "is_diverse": is_diverse,
            "has_variety": has_variety,
            "diagnosis": TestDiversityAnalyzer._diagnose_diversity(avg_similarity, has_variety),
            "pairwise_similarities": similarities
        }

    @staticmethod
    def _diagnose_diversity(avg_similarity: float, has_variety: bool) -> str:
        """Diagnose diversity issues."""
        if avg_similarity > 0.8:
            return "❌ CRITICAL: Recipes are nearly identical. Model checkpoint likely too early in training."
        elif avg_similarity > 0.7:
            return "⚠️ WARNING: Recipes are very similar. Consider using a later checkpoint or increasing temperature."
        elif avg_similarity > 0.5:
            return "⚠️ CAUTION: Moderate similarity detected. Model may benefit from more training."
        elif not has_variety:
            return "⚠️ CAUTION: Limited ingredient variety. Model may be overfitting."
        else:
            return "✅ GOOD: Recipes show healthy diversity."


def run_diversity_check(recipe_results: list):
    """
    Run diversity check on generated recipes.

    Args:
        recipe_results: List of recipe generation results
    """
    print("\n" + "=" * 80)
    print("RECIPE DIVERSITY ANALYSIS")
    print("=" * 80)

    recipes = [r.get('recipe', '') for r in recipe_results if r.get('recipe')]

    if len(recipes) < 2:
        print("⚠️ Need at least 2 recipes to analyze diversity")
        return

    analysis = TestDiversityAnalyzer.analyze_recipe_diversity(recipes)

    print(f"\n📊 Diversity Metrics:")
    print(f"   Average Similarity: {analysis['avg_similarity']:.3f}")
    print(f"   Average Lexical Diversity: {analysis['avg_lexical_diversity']:.3f}")
    print(f"   Unique Ingredients: {analysis['unique_ingredients']}")
    print(f"   Total Recipes: {analysis['total_recipes']}")
    print(f"\n{analysis['diagnosis']}")

    if analysis['avg_similarity'] > 0.7:
        print("\n💡 RECOMMENDATIONS:")
        print("   1. Use a later checkpoint (checkpoint 12000 may be too early)")
        print("   2. Increase generation temperature (try 0.9-1.2)")
        print("   3. Enable beam search with higher diversity penalty")
        print("   4. Use 'creative' mode instead of 'greedy' mode")
        print("   5. Check if model has trained on diverse enough data")

    print("\n" + "=" * 80)

    return analysis


if __name__ == '__main__':
    print("Running Recipe Generator Tests...\n")

    # Run unit tests
    unittest.main(argv=[''], verbosity=2, exit=False)

    print("\n" + "=" * 80)
    print("All tests completed!")
    print("=" * 80)
