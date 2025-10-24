#!/usr/bin/env python3
"""
Evaluation Metrics Module
Comprehensive metrics for recipe generation including BLEU, ROUGE, and custom metrics.
"""

import re
import math
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from dataclasses import dataclass, field

import torch
import numpy as np

# Try to import evaluation libraries
try:
    from rouge_score import rouge_scorer
    ROUGE_AVAILABLE = True
except ImportError:
    ROUGE_AVAILABLE = False
    print("⚠️  rouge-score not installed. Install with: pip install rouge-score")

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    import nltk
    BLEU_AVAILABLE = True
except ImportError:
    BLEU_AVAILABLE = False
    print("⚠️  NLTK not installed. Install with: pip install nltk")


@dataclass
class RecipeMetrics:
    """Container for all recipe evaluation metrics."""

    # Standard NLP metrics
    bleu_score: Optional[float] = None
    rouge1_f1: Optional[float] = None
    rouge2_f1: Optional[float] = None
    rougeL_f1: Optional[float] = None
    perplexity: Optional[float] = None

    # Recipe-specific metrics
    ingredient_coherence: Optional[float] = None
    instruction_quality: Optional[float] = None
    recipe_completeness: Optional[float] = None
    format_correctness: Optional[float] = None

    # Generation quality
    avg_length: Optional[float] = None
    diversity_score: Optional[float] = None

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            k: v for k, v in self.__dict__.items()
            if v is not None
        }

    def print_summary(self, prefix: str = "") -> None:
        """Print formatted summary of metrics."""
        print(f"\n{prefix}📊 Evaluation Metrics:")
        if self.bleu_score is not None:
            print(f"{prefix}   BLEU: {self.bleu_score:.4f}")
        if self.rouge1_f1 is not None:
            print(f"{prefix}   ROUGE-1 F1: {self.rouge1_f1:.4f}")
        if self.rouge2_f1 is not None:
            print(f"{prefix}   ROUGE-2 F1: {self.rouge2_f1:.4f}")
        if self.rougeL_f1 is not None:
            print(f"{prefix}   ROUGE-L F1: {self.rougeL_f1:.4f}")
        if self.perplexity is not None:
            print(f"{prefix}   Perplexity: {self.perplexity:.4f}")
        if self.ingredient_coherence is not None:
            print(f"{prefix}   Ingredient Coherence: {self.ingredient_coherence:.4f}")
        if self.instruction_quality is not None:
            print(f"{prefix}   Instruction Quality: {self.instruction_quality:.4f}")
        if self.recipe_completeness is not None:
            print(f"{prefix}   Completeness: {self.recipe_completeness:.4f}")


class BLEUCalculator:
    """Calculate BLEU scores for generated recipes."""

    def __init__(self):
        if not BLEU_AVAILABLE:
            raise ImportError("NLTK is required for BLEU calculation. Install with: pip install nltk")
        self.smoothing = SmoothingFunction()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        return text.lower().split()

    def calculate(self, reference: str, hypothesis: str) -> float:
        """Calculate BLEU score for a single reference/hypothesis pair."""
        ref_tokens = self.tokenize(reference)
        hyp_tokens = self.tokenize(hypothesis)

        # Use BLEU-4 with smoothing
        try:
            score = sentence_bleu(
                [ref_tokens],
                hyp_tokens,
                smoothing_function=self.smoothing.method1
            )
            return score
        except Exception as e:
            print(f"⚠️  BLEU calculation error: {e}")
            return 0.0

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> float:
        """Calculate average BLEU score for a batch."""
        if len(references) != len(hypotheses):
            raise ValueError("Number of references must match number of hypotheses")

        scores = [
            self.calculate(ref, hyp)
            for ref, hyp in zip(references, hypotheses)
        ]

        return sum(scores) / len(scores) if scores else 0.0


class ROUGECalculator:
    """Calculate ROUGE scores for generated recipes."""

    def __init__(self):
        if not ROUGE_AVAILABLE:
            raise ImportError("rouge-score is required. Install with: pip install rouge-score")
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def calculate(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """Calculate ROUGE scores for a single reference/hypothesis pair."""
        try:
            scores = self.scorer.score(reference, hypothesis)
            return {
                'rouge1_f1': scores['rouge1'].fmeasure,
                'rouge2_f1': scores['rouge2'].fmeasure,
                'rougeL_f1': scores['rougeL'].fmeasure,
            }
        except Exception as e:
            print(f"⚠️  ROUGE calculation error: {e}")
            return {'rouge1_f1': 0.0, 'rouge2_f1': 0.0, 'rougeL_f1': 0.0}

    def calculate_batch(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Calculate average ROUGE scores for a batch."""
        if len(references) != len(hypotheses):
            raise ValueError("Number of references must match number of hypotheses")

        all_scores = [
            self.calculate(ref, hyp)
            for ref, hyp in zip(references, hypotheses)
        ]

        # Average scores
        avg_scores = {
            'rouge1_f1': sum(s['rouge1_f1'] for s in all_scores) / len(all_scores),
            'rouge2_f1': sum(s['rouge2_f1'] for s in all_scores) / len(all_scores),
            'rougeL_f1': sum(s['rougeL_f1'] for s in all_scores) / len(all_scores),
        }

        return avg_scores


class RecipeQualityMetrics:
    """Calculate recipe-specific quality metrics."""

    @staticmethod
    def extract_ingredients(recipe_text: str) -> List[str]:
        """Extract ingredients from recipe text."""
        # Look for ingredients section
        ingredients = []
        in_ingredients = False

        for line in recipe_text.split('\n'):
            line = line.strip()

            if 'ingredient' in line.lower() and ':' in line:
                in_ingredients = True
                continue
            elif 'instruction' in line.lower() and ':' in line:
                in_ingredients = False
                break

            if in_ingredients and line.startswith('-'):
                ingredients.append(line[1:].strip())

        return ingredients

    @staticmethod
    def extract_instructions(recipe_text: str) -> List[str]:
        """Extract instructions from recipe text."""
        instructions = []
        in_instructions = False

        for line in recipe_text.split('\n'):
            line = line.strip()

            if 'instruction' in line.lower() and ':' in line:
                in_instructions = True
                continue

            if in_instructions and re.match(r'^\d+\.', line):
                instructions.append(line)

        return instructions

    def calculate_ingredient_coherence(self, recipe_text: str) -> float:
        """
        Calculate ingredient coherence score.
        Checks if ingredients are properly formatted and reasonable.
        """
        ingredients = self.extract_ingredients(recipe_text)

        if not ingredients:
            return 0.0

        score = 0.0
        total_checks = 0

        for ingredient in ingredients:
            # Check 1: Has quantity or amount (numbers or words like "cup", "tbsp")
            has_quantity = bool(re.search(r'\d+|cup|tablespoon|teaspoon|tbsp|tsp|pound|lb|ounce|oz', ingredient, re.I))
            score += 1 if has_quantity else 0
            total_checks += 1

            # Check 2: Reasonable length (not too short, not too long)
            length_ok = 10 <= len(ingredient) <= 100
            score += 1 if length_ok else 0
            total_checks += 1

            # Check 3: Contains actual food item (basic check)
            has_food_word = bool(re.search(r'[a-zA-Z]{3,}', ingredient))
            score += 1 if has_food_word else 0
            total_checks += 1

        return score / total_checks if total_checks > 0 else 0.0

    def calculate_instruction_quality(self, recipe_text: str) -> float:
        """
        Calculate instruction quality score.
        Checks if instructions are clear, numbered, and actionable.
        """
        instructions = self.extract_instructions(recipe_text)

        if not instructions:
            return 0.0

        score = 0.0
        total_checks = 0

        for instruction in instructions:
            # Check 1: Starts with number
            starts_with_number = bool(re.match(r'^\d+\.', instruction))
            score += 1 if starts_with_number else 0
            total_checks += 1

            # Check 2: Contains action verb
            action_verbs = ['add', 'mix', 'stir', 'cook', 'bake', 'heat', 'pour', 'combine',
                            'whisk', 'blend', 'cut', 'chop', 'dice', 'slice', 'serve']
            has_action = any(verb in instruction.lower() for verb in action_verbs)
            score += 1 if has_action else 0
            total_checks += 1

            # Check 3: Reasonable length
            length_ok = 20 <= len(instruction) <= 300
            score += 1 if length_ok else 0
            total_checks += 1

        return score / total_checks if total_checks > 0 else 0.0

    def calculate_completeness(self, recipe_text: str) -> float:
        """
        Calculate recipe completeness score.
        Checks if recipe has all essential sections.
        """
        score = 0.0
        total_checks = 4

        # Check 1: Has title
        lines = recipe_text.split('\n')
        has_title = len(lines) > 0 and len(lines[0].strip()) > 0
        score += 1 if has_title else 0

        # Check 2: Has ingredients section
        has_ingredients = 'ingredient' in recipe_text.lower()
        score += 1 if has_ingredients else 0

        # Check 3: Has instructions section
        has_instructions = 'instruction' in recipe_text.lower()
        score += 1 if has_instructions else 0

        # Check 4: Has multiple ingredients and instructions
        ingredients = self.extract_ingredients(recipe_text)
        instructions = self.extract_instructions(recipe_text)
        has_multiple = len(ingredients) >= 3 and len(instructions) >= 3
        score += 1 if has_multiple else 0

        return score / total_checks

    def calculate_format_correctness(self, recipe_text: str) -> float:
        """
        Calculate format correctness score.
        Checks if recipe follows the expected markdown format.
        """
        score = 0.0
        total_checks = 4

        # Check 1: Title is bold (has **)
        has_bold_title = recipe_text.strip().startswith('**')
        score += 1 if has_bold_title else 0

        # Check 2: Sections use proper headers
        has_proper_sections = bool(re.search(r'\*\*Ingredient', recipe_text, re.I))
        score += 1 if has_proper_sections else 0

        # Check 3: Ingredients use bullet points (-)
        has_bullet_ingredients = bool(re.search(r'^- ', recipe_text, re.M))
        score += 1 if has_bullet_ingredients else 0

        # Check 4: Instructions are numbered
        has_numbered_instructions = bool(re.search(r'^\d+\. ', recipe_text, re.M))
        score += 1 if has_numbered_instructions else 0

        return score / total_checks


class MetricsCalculator:
    """Main metrics calculator combining all metrics."""

    def __init__(self, compute_bleu: bool = True, compute_rouge: bool = True):
        self.compute_bleu = compute_bleu and BLEU_AVAILABLE
        self.compute_rouge = compute_rouge and ROUGE_AVAILABLE

        if self.compute_bleu:
            self.bleu_calculator = BLEUCalculator()
        if self.compute_rouge:
            self.rouge_calculator = ROUGECalculator()

        self.quality_metrics = RecipeQualityMetrics()

    def calculate_perplexity(self, loss: float) -> float:
        """Calculate perplexity from loss."""
        try:
            return math.exp(loss)
        except OverflowError:
            return float('inf')

    def calculate_all_metrics(
        self,
        references: List[str],
        hypotheses: List[str],
        loss: Optional[float] = None
    ) -> RecipeMetrics:
        """Calculate all metrics for a batch of recipes."""
        metrics = RecipeMetrics()

        # BLEU
        if self.compute_bleu:
            metrics.bleu_score = self.bleu_calculator.calculate_batch(references, hypotheses)

        # ROUGE
        if self.compute_rouge:
            rouge_scores = self.rouge_calculator.calculate_batch(references, hypotheses)
            metrics.rouge1_f1 = rouge_scores['rouge1_f1']
            metrics.rouge2_f1 = rouge_scores['rouge2_f1']
            metrics.rougeL_f1 = rouge_scores['rougeL_f1']

        # Perplexity
        if loss is not None:
            metrics.perplexity = self.calculate_perplexity(loss)

        # Recipe-specific metrics (average across hypotheses)
        ingredient_scores = []
        instruction_scores = []
        completeness_scores = []
        format_scores = []

        for hyp in hypotheses:
            ingredient_scores.append(self.quality_metrics.calculate_ingredient_coherence(hyp))
            instruction_scores.append(self.quality_metrics.calculate_instruction_quality(hyp))
            completeness_scores.append(self.quality_metrics.calculate_completeness(hyp))
            format_scores.append(self.quality_metrics.calculate_format_correctness(hyp))

        metrics.ingredient_coherence = sum(ingredient_scores) / len(ingredient_scores) if ingredient_scores else 0.0
        metrics.instruction_quality = sum(instruction_scores) / len(instruction_scores) if instruction_scores else 0.0
        metrics.recipe_completeness = sum(completeness_scores) / len(completeness_scores) if completeness_scores else 0.0
        metrics.format_correctness = sum(format_scores) / len(format_scores) if format_scores else 0.0

        # Generation quality
        metrics.avg_length = sum(len(hyp) for hyp in hypotheses) / len(hypotheses) if hypotheses else 0.0

        return metrics

    def calculate_single_metrics(self, reference: str, hypothesis: str) -> RecipeMetrics:
        """Calculate all metrics for a single recipe."""
        return self.calculate_all_metrics([reference], [hypothesis])


if __name__ == "__main__":
    # Test metrics
    print("🧪 Testing Metrics Calculator...")

    reference = """**Chocolate Chip Cookies**

**Ingredients:**
- 2 cups all-purpose flour
- 1 cup butter, softened
- 3/4 cup sugar
- 2 eggs
- 1 tsp vanilla extract
- 2 cups chocolate chips

**Instructions:**
1. Preheat oven to 375°F.
2. Mix butter and sugar until creamy.
3. Add eggs and vanilla, mix well.
4. Gradually add flour and mix.
5. Fold in chocolate chips.
6. Drop spoonfuls onto baking sheet.
7. Bake for 10-12 minutes until golden brown.
"""

    hypothesis = """**Chocolate Chip Cookies**

**Ingredients:**
- 2 cups flour
- 1 cup butter
- 1 cup sugar
- 2 eggs
- 1 teaspoon vanilla
- 2 cups chocolate chips

**Instructions:**
1. Preheat oven to 350°F.
2. Cream butter and sugar together.
3. Beat in eggs and vanilla.
4. Mix in flour gradually.
5. Stir in chocolate chips.
6. Place on cookie sheet.
7. Bake for 12 minutes.
"""

    calculator = MetricsCalculator(compute_bleu=True, compute_rouge=True)
    metrics = calculator.calculate_single_metrics(reference, hypothesis)

    metrics.print_summary()

    print("\n✅ Metrics test complete!")
