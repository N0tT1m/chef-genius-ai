#!/usr/bin/env python3
"""
Advanced Recipe Metrics Module
Provides ROUGE, BLEU, coherence checking, and other advanced metrics for recipe quality
"""

import re
from typing import Dict, List, Any, Optional, Set
from collections import Counter
import math


class RecipeMetrics:
    """Advanced metrics for recipe evaluation."""

    @staticmethod
    def calculate_bleu(generated: str, reference: str, n: int = 4) -> float:
        """
        Calculate BLEU score between generated and reference text.

        Args:
            generated: Generated recipe text
            reference: Reference recipe text
            n: Maximum n-gram size (default: 4)

        Returns:
            BLEU score (0.0 to 1.0)
        """
        def get_ngrams(text: str, n: int) -> List[tuple]:
            """Extract n-grams from text."""
            tokens = text.lower().split()
            return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

        def count_ngrams(ngrams: List[tuple]) -> Counter:
            """Count n-gram occurrences."""
            return Counter(ngrams)

        # Tokenize
        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()

        if len(gen_tokens) == 0 or len(ref_tokens) == 0:
            return 0.0

        # Brevity penalty
        bp = 1.0
        if len(gen_tokens) < len(ref_tokens):
            bp = math.exp(1 - len(ref_tokens) / len(gen_tokens))

        # Calculate precision for each n-gram size
        precisions = []
        for i in range(1, n + 1):
            gen_ngrams = get_ngrams(generated, i)
            ref_ngrams = get_ngrams(reference, i)

            if len(gen_ngrams) == 0:
                precisions.append(0.0)
                continue

            gen_counts = count_ngrams(gen_ngrams)
            ref_counts = count_ngrams(ref_ngrams)

            # Calculate clipped counts
            clipped_counts = 0
            total_counts = 0

            for ngram, count in gen_counts.items():
                clipped_counts += min(count, ref_counts.get(ngram, 0))
                total_counts += count

            precision = clipped_counts / total_counts if total_counts > 0 else 0.0
            precisions.append(precision)

        # Geometric mean of precisions
        if all(p > 0 for p in precisions):
            geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        else:
            geo_mean = 0.0

        bleu = bp * geo_mean
        return bleu

    @staticmethod
    def calculate_rouge(generated: str, reference: str, rouge_type: str = 'rouge-1') -> Dict[str, float]:
        """
        Calculate ROUGE score between generated and reference text.

        Args:
            generated: Generated recipe text
            reference: Reference recipe text
            rouge_type: Type of ROUGE ('rouge-1', 'rouge-2', 'rouge-l')

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        def get_ngrams(text: str, n: int) -> Set[tuple]:
            """Extract n-grams from text."""
            tokens = text.lower().split()
            return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))

        def get_lcs(s1: List[str], s2: List[str]) -> int:
            """Get length of longest common subsequence."""
            m, n = len(s1), len(s2)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if s1[i-1] == s2[j-1]:
                        dp[i][j] = dp[i-1][j-1] + 1
                    else:
                        dp[i][j] = max(dp[i-1][j], dp[i][j-1])

            return dp[m][n]

        gen_tokens = generated.lower().split()
        ref_tokens = reference.lower().split()

        if len(gen_tokens) == 0 or len(ref_tokens) == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        if rouge_type == 'rouge-1':
            n = 1
        elif rouge_type == 'rouge-2':
            n = 2
        elif rouge_type == 'rouge-l':
            # LCS-based ROUGE
            lcs_length = get_lcs(gen_tokens, ref_tokens)
            precision = lcs_length / len(gen_tokens) if len(gen_tokens) > 0 else 0.0
            recall = lcs_length / len(ref_tokens) if len(ref_tokens) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            return {"precision": precision, "recall": recall, "f1": f1}
        else:
            raise ValueError(f"Unknown ROUGE type: {rouge_type}")

        gen_ngrams = get_ngrams(generated, n)
        ref_ngrams = get_ngrams(reference, n)

        overlap = len(gen_ngrams & ref_ngrams)

        precision = overlap / len(gen_ngrams) if len(gen_ngrams) > 0 else 0.0
        recall = overlap / len(ref_ngrams) if len(ref_ngrams) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {"precision": precision, "recall": recall, "f1": f1}

    @staticmethod
    def check_coherence(recipe: str) -> Dict[str, Any]:
        """
        Check recipe coherence by validating ingredient-instruction alignment.

        Args:
            recipe: Recipe text to analyze

        Returns:
            Dictionary with coherence metrics
        """
        # Extract sections
        ingredients_section = ""
        instructions_section = ""

        lines = recipe.split('\n')
        current_section = None

        for line in lines:
            line_clean = line.strip()
            line_lower = line_clean.lower()

            if 'ingredients:' in line_lower:
                current_section = 'ingredients'
                continue
            elif any(x in line_lower for x in ['instructions:', 'directions:', 'steps:']):
                current_section = 'instructions'
                continue
            elif 'notes:' in line_lower:
                current_section = 'notes'
                continue

            if current_section == 'ingredients':
                ingredients_section += " " + line_clean
            elif current_section == 'instructions':
                instructions_section += " " + line_clean

        # Extract ingredient names
        ingredient_names = RecipeMetrics._extract_ingredient_names(ingredients_section)

        # Check which ingredients are mentioned in instructions
        instructions_lower = instructions_section.lower()
        mentioned_ingredients = []
        unmentioned_ingredients = []

        for ing in ingredient_names:
            if ing.lower() in instructions_lower:
                mentioned_ingredients.append(ing)
            else:
                unmentioned_ingredients.append(ing)

        # Calculate coherence score
        total_ingredients = len(ingredient_names)
        if total_ingredients == 0:
            coherence_score = 0.0
            ingredient_coverage = 0.0
        else:
            ingredient_coverage = len(mentioned_ingredients) / total_ingredients
            coherence_score = ingredient_coverage

        # Check for sequential flow in instructions
        sequential_markers = ['first', 'then', 'next', 'finally', 'after', 'before', 'while']
        has_sequential_flow = any(marker in instructions_lower for marker in sequential_markers)

        # Check for numbered steps
        has_numbered_steps = bool(re.search(r'\d+\.', instructions_section))

        # Structural coherence
        structural_score = sum([
            has_sequential_flow,
            has_numbered_steps,
            ingredient_coverage > 0.7
        ]) / 3.0

        overall_coherence = (coherence_score + structural_score) / 2.0

        return {
            "overall_coherence": overall_coherence,
            "ingredient_coverage": ingredient_coverage,
            "mentioned_ingredients": mentioned_ingredients,
            "unmentioned_ingredients": unmentioned_ingredients,
            "has_sequential_flow": has_sequential_flow,
            "has_numbered_steps": has_numbered_steps,
            "structural_score": structural_score,
            "total_ingredients": total_ingredients
        }

    @staticmethod
    def _extract_ingredient_names(ingredients_text: str) -> List[str]:
        """Extract ingredient names from ingredients section."""
        # Common measurement words to skip
        skip_words = {
            'cup', 'cups', 'tablespoon', 'tablespoons', 'teaspoon', 'teaspoons',
            'tbsp', 'tsp', 'oz', 'ounce', 'ounces', 'lb', 'lbs', 'pound', 'pounds',
            'gram', 'grams', 'kg', 'kilogram', 'ml', 'liter', 'liters',
            'pinch', 'dash', 'clove', 'cloves', 'piece', 'pieces',
            'small', 'medium', 'large', 'whole', 'fresh', 'dried', 'frozen',
            'chopped', 'diced', 'minced', 'sliced', 'grated', 'shredded',
            'to', 'taste', 'as', 'needed', 'optional', 'divided', 'of', 'the', 'a', 'an'
        }

        ingredients = []
        lines = ingredients_text.split('\n')

        for line in lines:
            # Remove list markers
            line = re.sub(r'^[-•*]\s*', '', line)
            line = re.sub(r'^\d+\.\s*', '', line)

            # Remove quantities
            line = re.sub(r'\d+/\d+', '', line)  # fractions
            line = re.sub(r'\d+\.?\d*', '', line)  # numbers

            # Extract words
            words = line.split()

            # Find the main ingredient (usually the first noun-like word)
            for i, word in enumerate(words):
                word_clean = word.strip('.,()[]').lower()

                if word_clean not in skip_words and len(word_clean) > 2:
                    # Take up to 2 words for compound ingredients
                    if i + 1 < len(words):
                        next_word = words[i + 1].strip('.,()[]').lower()
                        if next_word not in skip_words and len(next_word) > 2:
                            ingredients.append(f"{word_clean} {next_word}")
                        else:
                            ingredients.append(word_clean)
                    else:
                        ingredients.append(word_clean)
                    break

        return ingredients

    @staticmethod
    def calculate_lexical_diversity(text: str) -> float:
        """
        Calculate lexical diversity (Type-Token Ratio).

        Args:
            text: Text to analyze

        Returns:
            Lexical diversity score (0.0 to 1.0)
        """
        tokens = text.lower().split()
        if len(tokens) == 0:
            return 0.0

        unique_tokens = set(tokens)
        ttr = len(unique_tokens) / len(tokens)

        return ttr

    @staticmethod
    def calculate_readability(text: str) -> Dict[str, float]:
        """
        Calculate readability metrics (simplified Flesch Reading Ease).

        Args:
            text: Text to analyze

        Returns:
            Dictionary with readability scores
        """
        # Count sentences
        sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
        num_sentences = len(sentences)

        # Count words
        words = text.split()
        num_words = len(words)

        # Count syllables (simplified)
        def count_syllables(word: str) -> int:
            word = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            previous_was_vowel = False

            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = is_vowel

            # Adjust for silent e
            if word.endswith('e'):
                syllable_count -= 1

            # Minimum 1 syllable
            return max(1, syllable_count)

        num_syllables = sum(count_syllables(word) for word in words)

        if num_sentences == 0 or num_words == 0:
            return {
                "flesch_reading_ease": 0.0,
                "avg_words_per_sentence": 0.0,
                "avg_syllables_per_word": 0.0
            }

        # Calculate metrics
        avg_words_per_sentence = num_words / num_sentences
        avg_syllables_per_word = num_syllables / num_words

        # Flesch Reading Ease: 206.835 - 1.015(words/sentences) - 84.6(syllables/words)
        flesch = 206.835 - 1.015 * avg_words_per_sentence - 84.6 * avg_syllables_per_word
        flesch = max(0.0, min(100.0, flesch))  # Clamp between 0 and 100

        return {
            "flesch_reading_ease": flesch,
            "avg_words_per_sentence": avg_words_per_sentence,
            "avg_syllables_per_word": avg_syllables_per_word,
            "interpretation": RecipeMetrics._interpret_flesch(flesch)
        }

    @staticmethod
    def _interpret_flesch(score: float) -> str:
        """Interpret Flesch Reading Ease score."""
        if score >= 90:
            return "Very Easy"
        elif score >= 80:
            return "Easy"
        elif score >= 70:
            return "Fairly Easy"
        elif score >= 60:
            return "Standard"
        elif score >= 50:
            return "Fairly Difficult"
        elif score >= 30:
            return "Difficult"
        else:
            return "Very Difficult"

    @staticmethod
    def analyze_recipe_completeness(recipe: str) -> Dict[str, Any]:
        """
        Analyze recipe completeness and structure.

        Args:
            recipe: Recipe text to analyze

        Returns:
            Dictionary with completeness metrics
        """
        recipe_lower = recipe.lower()

        # Check for required sections
        has_title = bool(re.search(r'^[A-Z][^:]+$', recipe.split('\n')[0])) if recipe else False
        has_ingredients = 'ingredients:' in recipe_lower
        has_instructions = any(x in recipe_lower for x in ['instructions:', 'directions:', 'steps:'])
        has_notes = 'notes:' in recipe_lower or 'tips:' in recipe_lower

        # Check for metadata
        has_servings = any(word in recipe_lower for word in ['serves', 'servings', 'yield'])
        has_time = any(word in recipe_lower for word in ['minutes', 'hours', 'time', 'prep', 'cook'])
        has_temperature = bool(re.search(r'\d+\s*°?[FCfc]', recipe))

        # Count elements
        ingredient_count = len(re.findall(r'^\s*[-•*]\s*', recipe, re.MULTILINE))
        instruction_count = len(re.findall(r'^\s*\d+\.\s*', recipe, re.MULTILINE))

        # Completeness score
        section_score = sum([has_title, has_ingredients, has_instructions]) / 3.0
        metadata_score = sum([has_servings, has_time, has_temperature]) / 3.0
        content_score = min(1.0, (ingredient_count / 5 + instruction_count / 5) / 2)

        overall_completeness = (section_score * 0.4 + metadata_score * 0.3 + content_score * 0.3)

        return {
            "overall_completeness": overall_completeness,
            "has_title": has_title,
            "has_ingredients": has_ingredients,
            "has_instructions": has_instructions,
            "has_notes": has_notes,
            "has_servings": has_servings,
            "has_time": has_time,
            "has_temperature": has_temperature,
            "ingredient_count": ingredient_count,
            "instruction_count": instruction_count,
            "section_score": section_score,
            "metadata_score": metadata_score,
            "content_score": content_score
        }

    @staticmethod
    def comprehensive_evaluation(recipe: str, reference: Optional[str] = None) -> Dict[str, Any]:
        """
        Perform comprehensive evaluation of a recipe.

        Args:
            recipe: Generated recipe text
            reference: Optional reference recipe for comparison

        Returns:
            Dictionary with all evaluation metrics
        """
        evaluation = {
            "coherence": RecipeMetrics.check_coherence(recipe),
            "completeness": RecipeMetrics.analyze_recipe_completeness(recipe),
            "lexical_diversity": RecipeMetrics.calculate_lexical_diversity(recipe),
            "readability": RecipeMetrics.calculate_readability(recipe)
        }

        if reference:
            evaluation["bleu"] = RecipeMetrics.calculate_bleu(recipe, reference)
            evaluation["rouge_1"] = RecipeMetrics.calculate_rouge(recipe, reference, 'rouge-1')
            evaluation["rouge_2"] = RecipeMetrics.calculate_rouge(recipe, reference, 'rouge-2')
            evaluation["rouge_l"] = RecipeMetrics.calculate_rouge(recipe, reference, 'rouge-l')

        # Calculate overall score
        coherence_score = evaluation["coherence"]["overall_coherence"]
        completeness_score = evaluation["completeness"]["overall_completeness"]
        diversity_score = min(1.0, evaluation["lexical_diversity"] * 2)  # Scale up
        readability_score = evaluation["readability"]["flesch_reading_ease"] / 100.0

        evaluation["overall_score"] = (
            coherence_score * 0.30 +
            completeness_score * 0.30 +
            diversity_score * 0.20 +
            readability_score * 0.20
        )

        return evaluation
