#!/usr/bin/env python3
"""
Training Improvements Module
Contains all enhancement utilities for recipe model training:
- Validation and evaluation
- Advanced schedulers
- Curriculum learning
- Data augmentation
- Gradient noise
"""

import math
import random
import re
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from torch.optim.lr_scheduler import _LRScheduler


class WarmupCosineScheduler(_LRScheduler):
    """
    Cosine annealing with warmup and restarts.
    Combines linear warmup with cosine annealing for better convergence.
    """

    def __init__(self, optimizer, warmup_steps, T_0, T_mult=1, eta_min=1e-6, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.current_step = 0
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            alpha = self.current_step / max(1, self.warmup_steps)
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # Cosine annealing with restarts
            steps_since_warmup = self.current_step - self.warmup_steps

            # Calculate which cycle we're in
            cycle = 0
            cycle_length = self.T_0
            total_length = 0

            while total_length + cycle_length <= steps_since_warmup:
                total_length += cycle_length
                cycle += 1
                cycle_length = int(self.T_0 * (self.T_mult ** cycle))

            # Position within current cycle
            position = steps_since_warmup - total_length

            # Cosine annealing formula
            cos_inner = math.pi * position / cycle_length
            return [
                self.eta_min + (base_lr - self.eta_min) * (1 + math.cos(cos_inner)) / 2
                for base_lr in self.base_lrs
            ]

    def step(self, epoch=None):
        self.current_step += 1
        super().step(epoch)


class CurriculumManager:
    """
    Manages curriculum learning by controlling data complexity.
    Progressively introduces harder recipes during training.
    """

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0

    def get_difficulty_level(self, epoch: int) -> str:
        """Get difficulty level for current epoch."""
        progress = epoch / max(1, self.total_epochs)

        if progress < 0.3:  # First 30% - easy
            return 'easy'
        elif progress < 0.6:  # Next 30% - medium
            return 'medium'
        else:  # Last 40% - all difficulties
            return 'all'

    def get_quality_threshold(self, epoch: int) -> float:
        """Get minimum quality threshold for current epoch."""
        # Start at 0.5, gradually increase to 0.75
        progress = epoch / max(1, self.total_epochs)
        return 0.5 + (0.25 * progress)

    def get_max_complexity(self, epoch: int) -> Dict[str, int]:
        """Get maximum recipe complexity for filtering."""
        difficulty = self.get_difficulty_level(epoch)

        complexity_limits = {
            'easy': {'max_ingredients': 6, 'max_steps': 6},
            'medium': {'max_ingredients': 12, 'max_steps': 12},
            'all': {'max_ingredients': 999, 'max_steps': 999}
        }

        return complexity_limits[difficulty]


class RecipeAugmenter:
    """
    Augments recipe text for training diversity.
    Applies various transformations while preserving meaning.
    """

    def __init__(self, augmentation_probability: float = 0.3):
        self.aug_prob = augmentation_probability

        # Instruction paraphrase templates
        self.instruction_templates = [
            "Create a recipe: {text}",
            "Generate a detailed recipe: {text}",
            "Make a dish with these instructions: {text}",
            "Recipe for: {text}",
            "How to prepare: {text}",
            "Cooking instructions: {text}",
            "{text}",  # Original format
        ]

        # Cooking verbs for variation
        self.cooking_verbs = {
            'cook': ['prepare', 'make'],
            'mix': ['combine', 'blend', 'stir together'],
            'add': ['incorporate', 'include', 'put in'],
            'heat': ['warm', 'bring to temperature'],
            'chop': ['dice', 'cut', 'mince'],
        }

    def augment(self, text: str) -> str:
        """Apply random augmentation to recipe text."""
        if random.random() > self.aug_prob:
            return text

        # Choose augmentation strategy
        strategies = [
            self._vary_instruction_format,
            self._paraphrase_verbs,
            self._add_cooking_tips,
            lambda t: t  # No change (baseline)
        ]

        strategy = random.choice(strategies)
        return strategy(text)

    def _vary_instruction_format(self, text: str) -> str:
        """Vary the instruction format/template."""
        template = random.choice(self.instruction_templates)
        return template.format(text=text)

    def _paraphrase_verbs(self, text: str) -> str:
        """Replace cooking verbs with synonyms."""
        words = text.split()
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,!?')
            if word_lower in self.cooking_verbs:
                words[i] = random.choice(self.cooking_verbs[word_lower])
        return ' '.join(words)

    def _add_cooking_tips(self, text: str) -> str:
        """Add helpful cooking tips (10% probability)."""
        if random.random() > 0.1:
            return text

        tips = [
            "Tip: Taste and adjust seasonings as needed.",
            "Note: Fresh ingredients work best.",
            "Pro tip: Prepare all ingredients before starting.",
        ]

        return text + " " + random.choice(tips)


class GradientNoiseGenerator:
    """
    Adds annealed Gaussian noise to gradients.
    Helps escape local minima and improves generalization.
    """

    def __init__(self, eta: float = 0.3, gamma: float = 0.55):
        """
        Args:
            eta: Initial noise scale
            gamma: Annealing rate (higher = faster decay)
        """
        self.eta = eta
        self.gamma = gamma

    def add_noise(self, model: torch.nn.Module, step: int):
        """Add annealed noise to model gradients."""
        # Calculate noise variance (decreases over time)
        variance = self.eta / ((1 + step) ** self.gamma)

        if variance < 1e-8:  # Skip if noise too small
            return

        # Add noise to each parameter's gradient
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * math.sqrt(variance)
                param.grad.add_(noise)


class ValidationEvaluator:
    """
    Evaluates model on validation set.
    Tracks loss, perplexity, and generation quality.
    """

    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def evaluate(self, val_loader, max_batches: int = 100) -> Dict[str, float]:
        """
        Evaluate model on validation set.

        Args:
            val_loader: Validation data loader
            max_batches: Maximum batches to evaluate (for speed)

        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        total_loss = 0.0
        total_batches = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                if batch_idx >= max_batches:
                    break

                # Move batch to device
                batch = {
                    k: v.to(self.device, non_blocking=True) if hasattr(v, 'to') else v
                    for k, v in batch.items()
                }

                # Filter out non-model inputs
                model_inputs = {k: v for k, v in batch.items() if k != 'quality_scores'}

                # Forward pass
                outputs = self.model(**model_inputs)
                total_loss += outputs.loss.item()
                total_batches += 1

        # Calculate metrics
        avg_loss = total_loss / max(total_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))  # Cap to prevent overflow

        self.model.train()

        return {
            'val_loss': avg_loss,
            'val_perplexity': perplexity,
            'val_batches': total_batches
        }

    def generate_sample_recipes(self, prompts: List[str], max_length: int = 200) -> List[str]:
        """Generate sample recipes for qualitative evaluation."""
        self.model.eval()
        generated_recipes = []

        with torch.no_grad():
            for prompt in prompts:
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=128
                ).to(self.device)

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    num_beams=1,
                    pad_token_id=self.tokenizer.pad_token_id
                )

                recipe = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                generated_recipes.append(recipe)

        self.model.train()
        return generated_recipes


class RecipeQualityMetrics:
    """
    Evaluates recipe-specific quality metrics.
    Measures ingredient coherence, instruction quality, and completeness.
    """

    def __init__(self):
        # Common ingredient patterns
        self.ingredient_pattern = re.compile(r'\b\d+(?:/\d+)?\s*(?:cup|tbsp|tsp|oz|lb|g|kg|ml|l|pinch|dash)?\s+\w+')

        # Instruction action verbs
        self.action_verbs = {
            'mix', 'stir', 'combine', 'add', 'heat', 'cook', 'bake', 'boil',
            'fry', 'sauté', 'chop', 'dice', 'slice', 'pour', 'serve', 'place',
            'preheat', 'season', 'garnish', 'blend', 'whisk', 'fold'
        }

    def evaluate_recipe(self, recipe_text: str) -> Dict[str, float]:
        """
        Evaluate quality of generated recipe.

        Returns:
            Dictionary with quality scores (0-1 scale)
        """
        # Extract sections
        ingredients = self._extract_ingredients(recipe_text)
        instructions = self._extract_instructions(recipe_text)

        # Calculate metrics
        ingredient_coherence = self._score_ingredient_coherence(ingredients)
        instruction_quality = self._score_instruction_quality(instructions)
        completeness = self._score_completeness(recipe_text, ingredients, instructions)
        format_correctness = self._score_format(recipe_text)

        return {
            'ingredient_coherence': ingredient_coherence,
            'instruction_quality': instruction_quality,
            'completeness': completeness,
            'format_correctness': format_correctness,
            'overall_quality': (ingredient_coherence + instruction_quality + completeness + format_correctness) / 4
        }

    def _extract_ingredients(self, text: str) -> List[str]:
        """Extract ingredient list from recipe text."""
        # Look for ingredient section
        lines = text.split('\n')
        ingredients = []

        in_ingredients = False
        for line in lines:
            line_lower = line.lower().strip()

            if 'ingredient' in line_lower:
                in_ingredients = True
                continue

            if in_ingredients and ('instruction' in line_lower or 'direction' in line_lower or 'step' in line_lower):
                break

            if in_ingredients and line.strip():
                ingredients.append(line.strip())

        return ingredients

    def _extract_instructions(self, text: str) -> List[str]:
        """Extract instruction steps from recipe text."""
        lines = text.split('\n')
        instructions = []

        in_instructions = False
        for line in lines:
            line_lower = line.lower().strip()

            if 'instruction' in line_lower or 'direction' in line_lower or 'step' in line_lower:
                in_instructions = True
                continue

            if in_instructions and line.strip():
                instructions.append(line.strip())

        return instructions

    def _score_ingredient_coherence(self, ingredients: List[str]) -> float:
        """Score how well-formatted ingredients are."""
        if not ingredients:
            return 0.0

        # Check for quantity/unit patterns
        well_formatted = sum(1 for ing in ingredients if self.ingredient_pattern.search(ing.lower()))
        return well_formatted / len(ingredients)

    def _score_instruction_quality(self, instructions: List[str]) -> float:
        """Score instruction quality based on action verbs."""
        if not instructions:
            return 0.0

        # Check for action verbs in instructions
        instructions_with_verbs = 0
        for inst in instructions:
            inst_lower = inst.lower()
            if any(verb in inst_lower for verb in self.action_verbs):
                instructions_with_verbs += 1

        return instructions_with_verbs / len(instructions)

    def _score_completeness(self, text: str, ingredients: List[str], instructions: List[str]) -> float:
        """Score recipe completeness."""
        score = 0.0

        # Has ingredients section
        if ingredients:
            score += 0.33

        # Has instructions section
        if instructions:
            score += 0.33

        # Has reasonable length
        if 100 < len(text) < 2000:
            score += 0.34

        return score

    def _score_format(self, text: str) -> float:
        """Score format correctness."""
        score = 0.0
        text_lower = text.lower()

        # Has ingredient section header
        if 'ingredient' in text_lower:
            score += 0.5

        # Has instruction section header
        if any(word in text_lower for word in ['instruction', 'direction', 'step']):
            score += 0.5

        return score


# Label smoothing loss function
class LabelSmoothingLoss(torch.nn.Module):
    """
    Label smoothing loss for better calibration.
    Prevents model from becoming overconfident.
    """

    def __init__(self, smoothing: float = 0.1, ignore_index: int = -100):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Model predictions (batch_size, seq_len, vocab_size)
            labels: Ground truth labels (batch_size, seq_len)
        """
        vocab_size = logits.size(-1)

        # Reshape for computation
        logits = logits.view(-1, vocab_size)
        labels = labels.view(-1)

        # Create smoothed labels
        confidence = 1.0 - self.smoothing
        smooth_value = self.smoothing / (vocab_size - 1)

        # One-hot encoding with smoothing
        one_hot = torch.zeros_like(logits).fill_(smooth_value)
        one_hot.scatter_(1, labels.unsqueeze(1), confidence)

        # Mask padding tokens
        mask = (labels != self.ignore_index).float()
        one_hot = one_hot * mask.unsqueeze(1)

        # Compute loss
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        loss = -(one_hot * log_probs).sum(dim=-1)

        # Average over non-padding tokens
        return loss.sum() / mask.sum()
