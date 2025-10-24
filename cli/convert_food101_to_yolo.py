#!/usr/bin/env python3
"""
Convert Food-101 Dataset to YOLO Format
Prepares the dataset for training YOLOv8
"""

import os
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import random
import yaml

class Food101ToYOLO:
    """Convert Food-101 dataset to YOLO format."""

    def __init__(self, food101_dir, output_dir):
        self.food101_dir = Path(food101_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load class names
        self.class_names = self._load_class_names()
        print(f"📚 Found {len(self.class_names)} food categories")

    def _load_class_names(self):
        """Load food category names from Food-101."""
        # Food-101 uses directory names as class names
        images_dir = self.food101_dir / 'images'
        if images_dir.exists():
            return sorted([d.name for d in images_dir.iterdir() if d.is_dir()])
        return []

    def convert(self, train_val_split=0.8):
        """
        Convert Food-101 to YOLO format.

        Food-101 structure:
        food-101/
          images/
            apple_pie/
              image1.jpg
              image2.jpg
            ...

        YOLO structure:
        yolo_dataset/
          train/
            images/
            labels/
          val/
            images/
            labels/
          data.yaml
        """
        print(f"\n🔄 Converting Food-101 to YOLO format...")
        print(f"📂 Input: {self.food101_dir}")
        print(f"📂 Output: {self.output_dir}\n")

        # Create YOLO directory structure
        for split in ['train', 'val']:
            (self.output_dir / split / 'images').mkdir(parents=True, exist_ok=True)
            (self.output_dir / split / 'labels').mkdir(parents=True, exist_ok=True)

        # Process each food category
        total_images = 0
        train_count = 0
        val_count = 0

        for class_id, class_name in enumerate(tqdm(self.class_names, desc="Converting classes")):
            class_dir = self.food101_dir / 'images' / class_name
            if not class_dir.exists():
                continue

            # Get all images for this class
            images = list(class_dir.glob('*.jpg'))
            random.shuffle(images)

            # Split into train/val
            split_idx = int(len(images) * train_val_split)
            train_images = images[:split_idx]
            val_images = images[split_idx:]

            # Process training images
            for img_path in train_images:
                self._process_image(img_path, class_id, 'train')
                train_count += 1

            # Process validation images
            for img_path in val_images:
                self._process_image(img_path, class_id, 'val')
                val_count += 1

            total_images += len(images)

        print(f"\n✅ Conversion complete!")
        print(f"📊 Total images: {total_images}")
        print(f"   🚂 Training: {train_count}")
        print(f"   ✅ Validation: {val_count}")

        # Create data.yaml
        self._create_data_yaml()

        return self.output_dir

    def _process_image(self, img_path, class_id, split):
        """Process a single image and create YOLO label."""
        # Copy image
        dest_img = self.output_dir / split / 'images' / img_path.name
        shutil.copy2(img_path, dest_img)

        # Create YOLO label file
        # For Food-101, we treat each image as containing one centered food item
        label_path = self.output_dir / split / 'labels' / f"{img_path.stem}.txt"

        # Get image dimensions
        with Image.open(img_path) as img:
            width, height = img.size

        # YOLO format: class_id center_x center_y width height (normalized 0-1)
        # Assume food item takes up central 80% of image
        center_x = 0.5
        center_y = 0.5
        box_width = 0.8
        box_height = 0.8

        # Write label file
        with open(label_path, 'w') as f:
            f.write(f"{class_id} {center_x} {center_y} {box_width} {box_height}\n")

    def _create_data_yaml(self):
        """Create YOLO dataset configuration file."""
        config = {
            'path': str(self.output_dir.absolute()),
            'train': 'train/images',
            'val': 'val/images',
            'nc': len(self.class_names),  # Number of classes
            'names': self.class_names
        }

        yaml_path = self.output_dir / 'data.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n✅ Dataset config created: {yaml_path}")
        print(f"📊 Classes: {len(self.class_names)}")
        print(f"   Sample classes: {self.class_names[:5]}...")

        return yaml_path


def main():
    """Main conversion function."""
    import argparse

    parser = argparse.ArgumentParser(description='Convert Food-101 to YOLO format')
    parser.add_argument('--input', type=str, default='data/vision_training/food-101',
                        help='Path to Food-101 dataset directory')
    parser.add_argument('--output', type=str, default='data/vision_training/food101_yolo',
                        help='Output directory for YOLO format dataset')
    parser.add_argument('--split', type=float, default=0.8,
                        help='Train/val split ratio (default: 0.8)')

    args = parser.parse_args()

    # Check if input exists
    if not Path(args.input).exists():
        print(f"❌ Food-101 dataset not found: {args.input}")
        print("\n💡 Download it first:")
        print("   python cli/setup_vision_training.py")
        return

    # Convert dataset
    converter = Food101ToYOLO(args.input, args.output)
    output_path = converter.convert(train_val_split=args.split)

    print(f"\n✅ Dataset ready for training!")
    print(f"📁 Location: {output_path}")
    print(f"\n📝 Next step:")
    print(f"   python cli/train_food_detector.py --data {output_path}/data.yaml")


if __name__ == "__main__":
    main()
