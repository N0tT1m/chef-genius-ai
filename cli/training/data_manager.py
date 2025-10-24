#!/usr/bin/env python3
"""
Data Management Module
Handles dataset loading, splitting, and preprocessing with validation support.
"""

import os
import random
import json
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from transformers import PreTrainedTokenizer

from training.config import DataConfig


@dataclass
class DatasetSplitInfo:
    """Information about dataset splits."""
    train_size: int
    val_size: int
    total_size: int
    train_split_ratio: float

    def print_summary(self) -> None:
        """Print split information."""
        print(f"\n📊 Dataset Split Information:")
        print(f"   Total samples: {self.total_size:,}")
        print(f"   Train samples: {self.train_size:,} ({self.train_split_ratio:.1%})")
        print(f"   Val samples: {self.val_size:,} ({(1-self.train_split_ratio):.1%})")


class RecipeDataset(Dataset):
    """
    Recipe dataset with quality filtering and split support.
    Loads from validated JSONL files.
    """

    def __init__(
        self,
        jsonl_files: List[str],
        tokenizer: PreTrainedTokenizer,
        max_input_length: int = 256,
        max_target_length: int = 512,
        min_quality_score: float = 0.6,
    ):
        self.tokenizer = tokenizer
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.min_quality_score = min_quality_score

        # Load recipes
        self.recipes: List[Dict[str, Any]] = []
        self._load_recipes(jsonl_files)

    def _load_recipes(self, jsonl_files: List[str]) -> None:
        """Load recipes from JSONL files with quality filtering."""
        total_loaded = 0
        total_filtered = 0

        for jsonl_file in jsonl_files:
            if not os.path.exists(jsonl_file):
                print(f"⚠️  File not found: {jsonl_file}")
                continue

            print(f"📂 Loading recipes from: {jsonl_file}")
            file_count, file_filtered = self._load_jsonl_file(jsonl_file)
            total_loaded += file_count
            total_filtered += file_filtered
            print(f"   ✅ Loaded {file_count} recipes (filtered {file_filtered} low quality)")

        print(f"📊 Total loaded: {total_loaded} recipes")
        print(f"📊 Total in dataset: {len(self.recipes)} recipes (quality >= {self.min_quality_score})")

        if len(self.recipes) == 0:
            raise ValueError("No recipes loaded! Check your JSONL files.")

    def _load_jsonl_file(self, jsonl_file: str) -> Tuple[int, int]:
        """Load recipes from a single JSONL file."""
        loaded_count = 0
        filtered_count = 0

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        recipe_data = json.loads(line)

                        # Validate required fields
                        if not all(key in recipe_data for key in ['input', 'output', 'quality_score']):
                            continue

                        # Filter by quality score
                        if recipe_data['quality_score'] >= self.min_quality_score:
                            self.recipes.append({
                                'input': recipe_data['input'],
                                'output': recipe_data['output'],
                                'quality_score': recipe_data['quality_score'],
                                'source_file': os.path.basename(jsonl_file),
                                'line_number': line_num
                            })
                            loaded_count += 1
                        else:
                            filtered_count += 1

                    except json.JSONDecodeError as e:
                        print(f"⚠️  JSON error in {jsonl_file}:{line_num}: {e}")
                        continue

        except Exception as e:
            print(f"❌ Error loading {jsonl_file}: {e}")

        return loaded_count, filtered_count

    def __len__(self) -> int:
        return len(self.recipes)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        recipe = self.recipes[idx]

        # Tokenize input (instruction)
        input_encoding = self.tokenizer(
            recipe['input'],
            truncation=True,
            padding='max_length',
            max_length=self.max_input_length,
            return_tensors='pt'
        )

        # Tokenize target (recipe output)
        target_encoding = self.tokenizer(
            recipe['output'],
            truncation=True,
            padding='max_length',
            max_length=self.max_target_length,
            return_tensors='pt'
        )

        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'labels': target_encoding['input_ids'].squeeze(),
            'quality_score': torch.tensor(recipe['quality_score'], dtype=torch.float32),
            'source_file': recipe['source_file']
        }

    def shuffle(self) -> None:
        """Shuffle the dataset."""
        random.shuffle(self.recipes)

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics."""
        quality_scores = [recipe['quality_score'] for recipe in self.recipes]
        source_files: Dict[str, int] = {}

        for recipe in self.recipes:
            source = recipe['source_file']
            source_files[source] = source_files.get(source, 0) + 1

        return {
            'total_recipes': len(self.recipes),
            'avg_quality_score': sum(quality_scores) / len(quality_scores) if quality_scores else 0,
            'min_quality_score': min(quality_scores) if quality_scores else 0,
            'max_quality_score': max(quality_scores) if quality_scores else 0,
            'source_distribution': source_files
        }


class DataManager:
    """
    Manages dataset loading, splitting, and dataloader creation.
    Provides train/val splits and proper data handling.
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        config: DataConfig
    ):
        self.tokenizer = tokenizer
        self.config = config

        # Ensure pad token is set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Dataset will be loaded on demand
        self.full_dataset: Optional[RecipeDataset] = None
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.split_info: Optional[DatasetSplitInfo] = None

    def load_datasets(self) -> None:
        """Load and split datasets into train/val."""
        print("\n🚀 Loading datasets...")

        # Find JSONL files
        jsonl_files = self._find_training_files()
        if not jsonl_files:
            raise FileNotFoundError(f"No FLAN-T5 training files found in {self.config.validated_data_dir}")

        print(f"📁 Found {len(jsonl_files)} JSONL file(s)")

        # Load full dataset
        self.full_dataset = RecipeDataset(
            jsonl_files=jsonl_files,
            tokenizer=self.tokenizer,
            max_input_length=self.config.max_input_length,
            max_target_length=self.config.max_target_length,
            min_quality_score=self.config.min_quality_score
        )

        # Print stats
        stats = self.full_dataset.get_stats()
        print("\n📊 Dataset Statistics:")
        print(f"   Total recipes: {stats['total_recipes']:,}")
        print(f"   Avg quality: {stats['avg_quality_score']:.3f}")
        print(f"   Quality range: {stats['min_quality_score']:.3f} - {stats['max_quality_score']:.3f}")

        # Create train/val split
        self._create_splits()

        # Print split info
        if self.split_info:
            self.split_info.print_summary()

    def _find_training_files(self) -> List[str]:
        """Find all FLAN-T5 training JSONL files."""
        data_dir = self.config.validated_data_dir

        # Handle Docker context
        if not os.path.exists(data_dir) and os.path.exists(f"cli/{data_dir}"):
            data_dir = f"cli/{data_dir}"
            print(f"🐳 Docker context detected, using: {data_dir}")

        data_path = Path(data_dir)

        # Look for combined file first
        combined_file = data_path / "combined_all_datasets_flan_t5.jsonl"
        if combined_file.exists():
            return [str(combined_file)]

        # Look for specific patterns
        patterns = [
            "*_flan_t5.jsonl",
            "flan_t5_*.jsonl",
            "combined_*_flan_t5.jsonl"
        ]

        files = []
        for pattern in patterns:
            files.extend(data_path.glob(pattern))

        return [str(f) for f in files if f.is_file()]

    def _create_splits(self) -> None:
        """Create train/val splits from full dataset."""
        if self.full_dataset is None:
            raise ValueError("Dataset not loaded. Call load_datasets() first.")

        total_size = len(self.full_dataset)
        train_size = int(total_size * self.config.train_split)
        val_size = total_size - train_size

        # Create indices
        indices = list(range(total_size))
        if self.config.shuffle:
            random.shuffle(indices)

        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        # Create subsets
        self.train_dataset = Subset(self.full_dataset, train_indices)
        self.val_dataset = Subset(self.full_dataset, val_indices)

        # Store split info
        self.split_info = DatasetSplitInfo(
            train_size=train_size,
            val_size=val_size,
            total_size=total_size,
            train_split_ratio=self.config.train_split
        )

        print(f"✅ Created train/val split: {train_size:,} / {val_size:,}")

    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Custom collate function for FLAN-T5 format."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        quality_scores = torch.stack([item['quality_score'] for item in batch])

        # Replace padding tokens in labels with -100 (ignore in loss)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'quality_scores': quality_scores
        }

    def create_train_dataloader(self, batch_size: int) -> DataLoader:
        """Create training dataloader."""
        if self.train_dataset is None:
            raise ValueError("Train dataset not created. Call load_datasets() first.")

        return DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Disable multiprocessing to avoid pickling issues
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )

    def create_val_dataloader(self, batch_size: int) -> DataLoader:
        """Create validation dataloader."""
        if self.val_dataset is None:
            raise ValueError("Val dataset not created. Call load_datasets() first.")

        return DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,  # Don't shuffle validation
            num_workers=0,
            collate_fn=self._collate_fn,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=False
        )

    def get_train_size(self) -> int:
        """Get training dataset size."""
        if self.split_info is None:
            raise ValueError("Dataset not loaded.")
        return self.split_info.train_size

    def get_val_size(self) -> int:
        """Get validation dataset size."""
        if self.split_info is None:
            raise ValueError("Dataset not loaded.")
        return self.split_info.val_size

    def reset_for_epoch(self) -> None:
        """Reset/shuffle dataset for new epoch."""
        if self.full_dataset:
            self.full_dataset.shuffle()
            print("🔄 Dataset shuffled for next epoch")


if __name__ == "__main__":
    # Test the data manager
    from transformers import AutoTokenizer
    from training.config import DataConfig

    print("🧪 Testing DataManager...")

    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    config = DataConfig(
        validated_data_dir="./cli/validated_datasets",
        min_quality_score=0.7,
        train_split=0.9
    )

    manager = DataManager(tokenizer, config)
    manager.load_datasets()

    # Create dataloaders
    train_loader = manager.create_train_dataloader(batch_size=4)
    val_loader = manager.create_val_dataloader(batch_size=4)

    print(f"\n✅ Train batches: {len(train_loader)}")
    print(f"✅ Val batches: {len(val_loader)}")

    # Test one batch
    for batch in train_loader:
        print(f"\n📦 Sample batch:")
        print(f"   Input shape: {batch['input_ids'].shape}")
        print(f"   Labels shape: {batch['labels'].shape}")
        print(f"   Quality scores: {batch['quality_scores']}")
        break

    print("\n✅ DataManager test complete!")
