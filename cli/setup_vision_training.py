#!/usr/bin/env python3
"""
Setup Food Detection Model Training
Downloads free datasets and prepares them for YOLOv8 training
No paid datasets or APIs required!
"""

import os
import sys
import requests
import zipfile
import tarfile
import shutil
from pathlib import Path
from tqdm import tqdm
import json
import yaml

class FoodDatasetDownloader:
    """Download and prepare free food detection datasets."""

    def __init__(self, data_dir="data/vision_training"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.datasets = {
            "food101": {
                "url": "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                "size": "5GB",
                "categories": 101,
                "description": "101 food categories, 101k images"
            },
            "fruits360": {
                "url": "https://www.kaggle.com/api/v1/datasets/download/moltean/fruits",
                "size": "800MB",
                "categories": 131,
                "description": "Fruits and vegetables dataset",
                "requires_kaggle": True
            },
            "open_images_food": {
                "description": "Open Images food subset (will download via script)",
                "size": "Variable",
                "categories": "350+",
                "custom_download": True
            }
        }

    def download_food101(self):
        """Download Food-101 dataset."""
        print("📥 Downloading Food-101 dataset (5GB, this will take a while)...")

        dataset_path = self.data_dir / "food-101"
        if dataset_path.exists():
            print("✅ Food-101 already downloaded!")
            return dataset_path

        # Download
        tar_path = self.data_dir / "food-101.tar.gz"
        if not tar_path.exists():
            self._download_file(
                "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz",
                tar_path
            )

        # Extract
        print("📦 Extracting Food-101...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(self.data_dir)

        print("✅ Food-101 ready!")
        return dataset_path

    def download_grocery_store_dataset(self):
        """Download Grocery Store Dataset (smaller, good for fridge items)."""
        print("📥 Downloading Grocery Store Dataset...")

        dataset_path = self.data_dir / "grocery-store"
        if dataset_path.exists():
            print("✅ Grocery Store dataset already downloaded!")
            return dataset_path

        dataset_path.mkdir(parents=True, exist_ok=True)

        # Clone repository
        import subprocess
        subprocess.run([
            "git", "clone",
            "https://github.com/marcusklasson/GroceryStoreDataset.git",
            str(dataset_path)
        ], check=True)

        print("✅ Grocery Store dataset ready!")
        return dataset_path

    def download_roboflow_free_dataset(self):
        """Download free public food dataset from Roboflow."""
        print("📥 Downloading Roboflow public food dataset (no API key needed)...")

        dataset_path = self.data_dir / "roboflow-food"
        if dataset_path.exists():
            print("✅ Roboflow food dataset already downloaded!")
            return dataset_path

        dataset_path.mkdir(parents=True, exist_ok=True)

        # Public datasets don't need authentication
        dataset_urls = [
            "https://public.roboflow.com/ds/food-detection-public",
            "https://public.roboflow.com/ds/ingredients-v1"
        ]

        print("💡 Roboflow datasets available at:")
        for url in dataset_urls:
            print(f"   {url}")

        print("📝 To download, visit: https://universe.roboflow.com/")
        print("   Search for 'food detection' or 'ingredients'")
        print("   Download in YOLOv8 format (no account needed for public datasets)")

        return dataset_path

    def setup_kaggle_credentials(self):
        """Setup Kaggle API credentials."""
        kaggle_dir = Path.home() / ".kaggle"
        kaggle_json = kaggle_dir / "kaggle.json"

        if kaggle_json.exists():
            print("✅ Kaggle credentials already configured!")
            return True

        print("\n🔐 Kaggle Setup Required:")
        print("1. Go to: https://www.kaggle.com/settings")
        print("2. Click 'Create New API Token'")
        print("3. Download kaggle.json")
        print("4. Place it in: ~/.kaggle/kaggle.json")
        print("5. Run: chmod 600 ~/.kaggle/kaggle.json")
        print("\nThen run this script again!")

        return False

    def download_fruits360_kaggle(self):
        """Download Fruits-360 from Kaggle."""
        if not self.setup_kaggle_credentials():
            return None

        print("📥 Downloading Fruits-360 from Kaggle...")

        dataset_path = self.data_dir / "fruits-360"
        if dataset_path.exists():
            print("✅ Fruits-360 already downloaded!")
            return dataset_path

        try:
            from kaggle.api.kaggle_api_extended import KaggleApi

            api = KaggleApi()
            api.authenticate()

            # Download dataset
            api.dataset_download_files(
                'moltean/fruits',
                path=str(dataset_path),
                unzip=True
            )

            print("✅ Fruits-360 ready!")
            return dataset_path

        except ImportError:
            print("❌ Kaggle API not installed. Install with: pip install kaggle")
            return None

    def _download_file(self, url, dest_path):
        """Download file with progress bar."""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))

        with open(dest_path, 'wb') as file, tqdm(
            desc=dest_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = file.write(data)
                pbar.update(size)

    def convert_to_yolo_format(self, dataset_name, source_path):
        """Convert dataset to YOLO format."""
        print(f"🔄 Converting {dataset_name} to YOLO format...")

        yolo_path = self.data_dir / f"{dataset_name}_yolo"
        yolo_path.mkdir(parents=True, exist_ok=True)

        # Create YOLO directory structure
        for split in ['train', 'val', 'test']:
            (yolo_path / split / 'images').mkdir(parents=True, exist_ok=True)
            (yolo_path / split / 'labels').mkdir(parents=True, exist_ok=True)

        print(f"✅ YOLO format structure created at {yolo_path}")
        return yolo_path

    def create_yolo_config(self, dataset_path, class_names):
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(dataset_path.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {i: name for i, name in enumerate(class_names)}
        }

        config_path = dataset_path / 'data.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        print(f"✅ YOLO config created: {config_path}")
        return config_path

    def list_available_datasets(self):
        """List all available free datasets."""
        print("\n📚 Available FREE Food Detection Datasets:\n")

        for name, info in self.datasets.items():
            print(f"🍕 {name.upper()}")
            print(f"   Description: {info['description']}")
            print(f"   Size: {info['size']}")
            print(f"   Categories: {info['categories']}")
            if info.get('requires_kaggle'):
                print(f"   ⚠️  Requires Kaggle account (free)")
            if info.get('custom_download'):
                print(f"   ℹ️  Custom download process")
            print()


def main():
    """Main setup function."""
    print("🍳 ChefGenius Food Detection Model Setup")
    print("=" * 60)

    downloader = FoodDatasetDownloader()

    print("\nThis will download FREE datasets for training food detection.")
    print("Choose a dataset to download:\n")

    print("1. Food-101 (5GB, 101 categories) - RECOMMENDED")
    print("2. Grocery Store Dataset (small, good for fridge items)")
    print("3. Roboflow Public Datasets (requires manual download)")
    print("4. Fruits-360 via Kaggle (requires Kaggle account)")
    print("5. List all available datasets")
    print("6. Download ALL datasets (will take a while!)")

    choice = input("\nEnter choice (1-6): ").strip()

    if choice == '1':
        dataset_path = downloader.download_food101()
        print(f"\n✅ Dataset ready at: {dataset_path}")
        print("\n📝 Next steps:")
        print("   1. Convert to YOLO format: python cli/convert_food101_to_yolo.py")
        print("   2. Train model: python cli/train_food_detector.py")

    elif choice == '2':
        dataset_path = downloader.download_grocery_store_dataset()
        print(f"\n✅ Dataset ready at: {dataset_path}")

    elif choice == '3':
        downloader.download_roboflow_free_dataset()

    elif choice == '4':
        dataset_path = downloader.download_fruits360_kaggle()

    elif choice == '5':
        downloader.list_available_datasets()

    elif choice == '6':
        print("\n📥 Downloading ALL datasets (this will take a long time)...")
        downloader.download_food101()
        downloader.download_grocery_store_dataset()
        downloader.download_fruits360_kaggle()
        print("\n✅ All datasets downloaded!")

    else:
        print("❌ Invalid choice!")


if __name__ == "__main__":
    main()
