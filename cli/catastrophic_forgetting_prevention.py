#!/usr/bin/env python3
"""
Catastrophic Forgetting Prevention for Recipe Model
Implements techniques to maintain performance on old recipes while learning new ones
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import Dataset
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EWCRegularizer:
    """Elastic Weight Consolidation for preventing catastrophic forgetting."""
    
    def __init__(self, model: GPT2LMHeadModel, importance_dataset: Dataset, tokenizer: GPT2Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device
        
        # Compute Fisher Information Matrix
        self.fisher_information = self._compute_fisher_information(importance_dataset)
        
        # Store optimal parameters (current model state)
        self.optimal_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.optimal_params[name] = param.data.clone()
                
    def _compute_fisher_information(self, dataset: Dataset, num_samples: int = 1000) -> Dict[str, torch.Tensor]:
        """Compute Fisher Information Matrix for important parameters."""
        logger.info("Computing Fisher Information Matrix...")
        
        self.model.eval()
        fisher_info = {}
        
        # Initialize Fisher information dict
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_info[name] = torch.zeros_like(param.data)
                
        # Sample from dataset and compute gradients
        sample_size = min(num_samples, len(dataset))
        indices = random.sample(range(len(dataset)), sample_size)
        
        for i, idx in enumerate(indices):
            if i % 100 == 0:
                logger.info(f"Processing sample {i}/{sample_size}")
                
            # Get data sample
            sample = dataset[idx]
            input_ids = torch.tensor(sample['input_ids']).unsqueeze(0).to(self.device)
            labels = torch.tensor(sample['labels']).unsqueeze(0).to(self.device)
            
            # Forward pass
            self.model.zero_grad()
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
            
            # Backward pass
            loss.backward()
            
            # Accumulate Fisher information
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_info[name] += param.grad.data ** 2
                    
        # Normalize by number of samples
        for name in fisher_info:
            fisher_info[name] /= sample_size
            
        logger.info("Fisher Information Matrix computation completed")
        return fisher_info
        
    def get_ewc_loss(self, lambda_ewc: float = 1000.0) -> torch.Tensor:
        """Compute EWC regularization loss."""
        ewc_loss = 0
        
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.fisher_information:
                fisher = self.fisher_information[name]
                optimal = self.optimal_params[name]
                ewc_loss += (fisher * (param - optimal) ** 2).sum()
                
        return lambda_ewc * ewc_loss

class ReplayBuffer:
    """Memory buffer for storing representative examples from old tasks."""
    
    def __init__(self, max_size: int = 1000, selection_strategy: str = "random"):
        self.max_size = max_size
        self.selection_strategy = selection_strategy
        self.buffer = []
        self.embeddings = []
        
    def add_samples(self, samples: List[Dict], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
        """Add samples to replay buffer with intelligent selection."""
        
        if self.selection_strategy == "diverse":
            samples = self._select_diverse_samples(samples, model, tokenizer)
        elif self.selection_strategy == "random":
            samples = random.sample(samples, min(len(samples), self.max_size // 2))
            
        # Add to buffer
        self.buffer.extend(samples)
        
        # Trim buffer if it exceeds max size
        if len(self.buffer) > self.max_size:
            if self.selection_strategy == "diverse":
                self._trim_buffer_diverse(model, tokenizer)
            else:
                # Remove oldest samples
                self.buffer = self.buffer[-self.max_size:]
                
        logger.info(f"Replay buffer updated. Current size: {len(self.buffer)}")
        
    def _select_diverse_samples(self, samples: List[Dict], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> List[Dict]:
        """Select diverse samples using embedding similarity."""
        if len(samples) <= self.max_size // 2:
            return samples
            
        # Get embeddings for samples
        embeddings = self._get_embeddings(samples, model, tokenizer)
        
        # Use k-means clustering or greedy selection for diversity
        selected_indices = self._greedy_diverse_selection(embeddings, self.max_size // 2)
        
        return [samples[i] for i in selected_indices]
        
    def _get_embeddings(self, samples: List[Dict], model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer) -> np.ndarray:
        """Get embeddings for samples."""
        model.eval()
        embeddings = []
        
        with torch.no_grad():
            for sample in samples:
                # Format sample
                text = self._format_sample(sample)
                
                # Tokenize
                inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
                
                # Get hidden states
                outputs = model(**inputs, output_hidden_states=True)
                
                # Use mean of last hidden state as embedding
                embedding = outputs.hidden_states[-1].mean(dim=1).squeeze().cpu().numpy()
                embeddings.append(embedding)
                
        return np.array(embeddings)
        
    def _format_sample(self, sample: Dict) -> str:
        """Format sample for embedding computation."""
        if 'text' in sample:
            return sample['text']
        elif isinstance(sample, dict) and 'title' in sample:
            # Recipe format
            ingredients = " | ".join(sample.get('ingredients', []))
            return f"<TITLE>{sample['title']}</TITLE><INGREDIENTS>{ingredients}</INGREDIENTS><INSTRUCTIONS>{sample.get('instructions', '')}</INSTRUCTIONS>"
        else:
            return str(sample)
            
    def _greedy_diverse_selection(self, embeddings: np.ndarray, k: int) -> List[int]:
        """Greedy selection for diverse samples."""
        selected = []
        remaining = list(range(len(embeddings)))
        
        # Select first sample randomly
        first_idx = random.choice(remaining)
        selected.append(first_idx)
        remaining.remove(first_idx)
        
        # Greedily select most diverse samples
        for _ in range(k - 1):
            if not remaining:
                break
                
            max_min_distance = -1
            best_idx = None
            
            for idx in remaining:
                # Compute minimum distance to selected samples
                min_distance = min([
                    1 - cosine_similarity([embeddings[idx]], [embeddings[sel_idx]])[0][0]
                    for sel_idx in selected
                ])
                
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    best_idx = idx
                    
            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)
                
        return selected
        
    def _trim_buffer_diverse(self, model: GPT2LMHeadModel, tokenizer: GPT2Tokenizer):
        """Trim buffer maintaining diversity."""
        if len(self.buffer) <= self.max_size:
            return
            
        # Get embeddings for current buffer
        embeddings = self._get_embeddings(self.buffer, model, tokenizer)
        
        # Select diverse subset
        selected_indices = self._greedy_diverse_selection(embeddings, self.max_size)
        self.buffer = [self.buffer[i] for i in selected_indices]
        
    def get_replay_samples(self, num_samples: int) -> List[Dict]:
        """Get samples for replay."""
        if len(self.buffer) <= num_samples:
            return self.buffer.copy()
        return random.sample(self.buffer, num_samples)

class ContinualLearningTrainer:
    """Main trainer with catastrophic forgetting prevention."""
    
    def __init__(self, 
                 model_path: str, 
                 forgetting_prevention_method: str = "ewc",
                 replay_buffer_size: int = 1000,
                 ewc_lambda: float = 1000.0):
        
        self.model_path = Path(model_path)
        self.forgetting_method = forgetting_prevention_method
        self.ewc_lambda = ewc_lambda
        
        # Load model and tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        self.model = GPT2LMHeadModel.from_pretrained(model_path)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Initialize components
        self.ewc_regularizer = None
        self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size, selection_strategy="diverse")
        
        # Performance tracking
        self.performance_history = []
        
    def setup_ewc(self, importance_dataset: Dataset):
        """Setup EWC regularizer with importance dataset."""
        logger.info("Setting up EWC regularizer...")
        self.ewc_regularizer = EWCRegularizer(self.model, importance_dataset, self.tokenizer)
        
    def add_to_replay_buffer(self, samples: List[Dict]):
        """Add samples to replay buffer."""
        self.replay_buffer.add_samples(samples, self.model, self.tokenizer)
        
    def prepare_continual_dataset(self, 
                                 new_samples: List[Dict], 
                                 replay_ratio: float = 0.3) -> Dataset:
        """Prepare dataset mixing new samples with replay samples."""
        
        # Get replay samples
        replay_samples = []
        if replay_ratio > 0:
            num_replay = int(len(new_samples) * replay_ratio)
            replay_samples = self.replay_buffer.get_replay_samples(num_replay)
            
        # Combine samples
        all_samples = new_samples + replay_samples
        random.shuffle(all_samples)
        
        # Format and tokenize
        texts = []
        for sample in all_samples:
            if isinstance(sample, dict) and 'title' in sample:
                # Recipe format
                ingredients = " | ".join(sample.get('ingredients', []))
                text = f"<TITLE>{sample['title']}</TITLE><INGREDIENTS>{ingredients}</INGREDIENTS><INSTRUCTIONS>{sample.get('instructions', '')}</INSTRUCTIONS>"
            else:
                text = sample.get('text', str(sample))
            texts.append(text)
            
        # Tokenize
        tokenized = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        dataset = Dataset.from_dict({
            'input_ids': tokenized['input_ids'],
            'attention_mask': tokenized['attention_mask'],
            'labels': tokenized['input_ids'].clone()
        })
        
        logger.info(f"Prepared continual learning dataset: {len(new_samples)} new + {len(replay_samples)} replay = {len(all_samples)} total")
        return dataset
        
    def compute_continual_loss(self, batch, base_loss: torch.Tensor) -> torch.Tensor:
        """Compute loss with catastrophic forgetting prevention."""
        total_loss = base_loss
        
        if self.forgetting_method == "ewc" and self.ewc_regularizer is not None:
            ewc_loss = self.ewc_regularizer.get_ewc_loss(self.ewc_lambda)
            total_loss += ewc_loss
            
        return total_loss
        
    def evaluate_on_old_tasks(self, eval_datasets: List[Dataset]) -> Dict[str, float]:
        """Evaluate model performance on previous tasks."""
        self.model.eval()
        results = {}
        
        with torch.no_grad():
            for i, dataset in enumerate(eval_datasets):
                total_loss = 0
                num_samples = 0
                
                for j in range(min(len(dataset), 100)):  # Sample for efficiency
                    sample = dataset[j]
                    input_ids = torch.tensor(sample['input_ids']).unsqueeze(0)
                    labels = torch.tensor(sample['labels']).unsqueeze(0)
                    
                    outputs = self.model(input_ids=input_ids, labels=labels)
                    total_loss += outputs.loss.item()
                    num_samples += 1
                    
                avg_loss = total_loss / num_samples if num_samples > 0 else float('inf')
                results[f'task_{i}_loss'] = avg_loss
                results[f'task_{i}_perplexity'] = np.exp(avg_loss)
                
        return results
        
    def track_performance(self, eval_results: Dict[str, float], task_name: str = ""):
        """Track performance over time."""
        entry = {
            'timestamp': str(torch.cuda.current_stream().cuda_stream if torch.cuda.is_available() else 'cpu'),
            'task_name': task_name,
            'results': eval_results
        }
        self.performance_history.append(entry)
        
        # Save performance history
        history_file = self.model_path.parent / "performance_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.performance_history, f, indent=2)
            
    def save_continual_state(self, output_path: str):
        """Save model state and continual learning components."""
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save EWC state if available
        if self.ewc_regularizer is not None:
            ewc_state = {
                'fisher_information': {name: tensor.cpu().numpy().tolist() 
                                     for name, tensor in self.ewc_regularizer.fisher_information.items()},
                'optimal_params': {name: tensor.cpu().numpy().tolist() 
                                 for name, tensor in self.ewc_regularizer.optimal_params.items()}
            }
            with open(output_path / "ewc_state.json", 'w') as f:
                json.dump(ewc_state, f)
                
        # Save replay buffer
        if self.replay_buffer.buffer:
            with open(output_path / "replay_buffer.json", 'w') as f:
                json.dump(self.replay_buffer.buffer, f)
                
        # Save performance history
        with open(output_path / "performance_history.json", 'w') as f:
            json.dump(self.performance_history, f, indent=2)
            
        logger.info(f"Continual learning state saved to {output_path}")

def load_previous_training_data(data_cache_path: str, sample_size: int = 500) -> List[Dict]:
    """Load sample of previous training data for EWC computation."""
    cache_path = Path(data_cache_path)
    
    if not cache_path.exists():
        logger.warning(f"Training data cache not found: {cache_path}")
        return []
        
    try:
        with open(cache_path, 'r') as f:
            cached_data = json.load(f)
            
        # Convert to recipe format if needed
        samples = []
        for item in cached_data[:sample_size]:
            if isinstance(item, str):
                # Parse formatted text back to recipe
                sample = {'text': item}
            else:
                sample = item
            samples.append(sample)
            
        logger.info(f"Loaded {len(samples)} samples from training cache")
        return samples
        
    except Exception as e:
        logger.error(f"Error loading training cache: {e}")
        return []

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Continual learning with catastrophic forgetting prevention")
    parser.add_argument("--model-path", required=True, help="Path to base model")
    parser.add_argument("--new-data", required=True, help="Path to new training data (JSON/CSV)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--method", choices=["ewc", "replay", "both"], default="both", 
                       help="Forgetting prevention method")
    parser.add_argument("--ewc-lambda", type=float, default=1000.0, help="EWC regularization strength")
    parser.add_argument("--replay-buffer-size", type=int, default=1000, help="Replay buffer size")
    parser.add_argument("--replay-ratio", type=float, default=0.3, help="Ratio of replay samples in training")
    parser.add_argument("--cache-path", help="Path to training data cache for EWC")
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ContinualLearningTrainer(
        args.model_path,
        forgetting_prevention_method=args.method,
        replay_buffer_size=args.replay_buffer_size,
        ewc_lambda=args.ewc_lambda
    )
    
    # Load new data
    if args.new_data.endswith('.csv'):
        from incremental_training import load_new_recipes_from_csv
        new_samples = load_new_recipes_from_csv(args.new_data)
    else:
        with open(args.new_data, 'r') as f:
            new_samples = json.load(f)
            
    logger.info(f"Loaded {len(new_samples)} new samples")
    
    # Setup EWC if needed
    if args.method in ["ewc", "both"] and args.cache_path:
        previous_samples = load_previous_training_data(args.cache_path)
        if previous_samples:
            # Convert to dataset for EWC
            texts = [trainer.replay_buffer._format_sample(s) for s in previous_samples]
            tokenized = trainer.tokenizer(texts, truncation=True, padding=True, max_length=512, return_tensors='pt')
            importance_dataset = Dataset.from_dict({
                'input_ids': tokenized['input_ids'],
                'attention_mask': tokenized['attention_mask'],
                'labels': tokenized['input_ids'].clone()
            })
            trainer.setup_ewc(importance_dataset)
            
    # Add samples to replay buffer if using replay
    if args.method in ["replay", "both"]:
        if args.cache_path:
            previous_samples = load_previous_training_data(args.cache_path)
            trainer.add_to_replay_buffer(previous_samples)
            
    # Prepare continual learning dataset
    continual_dataset = trainer.prepare_continual_dataset(new_samples, args.replay_ratio)
    
    # Training would happen here with continual_dataset
    # This example shows the setup - actual training loop would use
    # trainer.compute_continual_loss() in the training step
    
    # Save final state
    trainer.save_continual_state(args.output_dir)
    
    logger.info("Continual learning setup completed")

if __name__ == "__main__":
    main()