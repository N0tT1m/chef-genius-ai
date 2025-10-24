#!/usr/bin/env python3
"""
Train YOLOv8 Food Detection Model on RTX 5090
Optimized for maximum performance on your GPU
"""

import os
import sys
import torch
from pathlib import Path
from ultralytics import YOLO
import wandb
from datetime import datetime

class FoodDetectorTrainer:
    """Train food detection model optimized for RTX 5090."""

    def __init__(
        self,
        model_size='x',  # yolov8x for RTX 5090 (largest model)
        data_yaml='data/vision_training/food_dataset.yaml',
        output_dir='models/food_detector'
    ):
        self.model_size = model_size
        self.data_yaml = data_yaml
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # RTX 5090 optimizations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.gpu_info = self._get_gpu_info()

        print(f"🚀 Training on: {self.gpu_info}")

    def _get_gpu_info(self):
        """Get GPU information."""
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            return f"{gpu_name} ({gpu_memory:.1f}GB VRAM)"
        return "CPU (No CUDA available)"

    def train_from_scratch(
        self,
        epochs=100,
        batch_size=32,  # RTX 5090 can handle large batches
        image_size=640,
        workers=16,  # More workers for faster data loading
        use_wandb=True,
        resume=False
    ):
        """
        Train YOLOv8 food detection model.

        RTX 5090 Optimizations:
        - Large batch size (32)
        - High image resolution (640)
        - Many workers (16) for data loading
        - Mixed precision training (automatic)
        - Cache dataset in RAM for speed
        """

        print("\n" + "="*60)
        print("🍕 FOOD DETECTION MODEL TRAINING")
        print("="*60)
        print(f"📊 Model: YOLOv8{self.model_size}")
        print(f"📁 Dataset: {self.data_yaml}")
        print(f"🎯 Epochs: {epochs}")
        print(f"📦 Batch Size: {batch_size}")
        print(f"🖼️  Image Size: {image_size}x{image_size}")
        print(f"💻 Device: {self.gpu_info}")
        print(f"👷 Workers: {workers}")
        print("="*60 + "\n")

        # Initialize W&B for tracking
        if use_wandb:
            wandb.init(
                project="chefgenius-food-detection",
                name=f"yolov8{self.model_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config={
                    "model": f"yolov8{self.model_size}",
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "image_size": image_size,
                    "gpu": self.gpu_info
                }
            )

        # Load model
        model_name = f'yolov8{self.model_size}.pt'
        print(f"📥 Loading {model_name}...")

        if resume and (self.output_dir / 'weights' / 'last.pt').exists():
            print("♻️  Resuming from last checkpoint...")
            model = YOLO(self.output_dir / 'weights' / 'last.pt')
        else:
            print("🆕 Starting fresh training...")
            model = YOLO(model_name)

        # RTX 5090 Training Configuration
        training_args = {
            'data': self.data_yaml,
            'epochs': epochs,
            'batch': batch_size,
            'imgsz': image_size,
            'device': 0,  # Use first GPU
            'workers': workers,
            'project': str(self.output_dir),
            'name': 'train',
            'exist_ok': True,

            # Performance optimizations
            'cache': 'ram',  # Cache entire dataset in RAM (fast!)
            'amp': True,  # Automatic mixed precision (faster)
            'verbose': True,

            # Data augmentation
            'hsv_h': 0.015,  # HSV-Hue augmentation
            'hsv_s': 0.7,    # HSV-Saturation augmentation
            'hsv_v': 0.4,    # HSV-Value augmentation
            'degrees': 10,   # Rotation augmentation
            'translate': 0.1, # Translation augmentation
            'scale': 0.5,    # Scale augmentation
            'flipud': 0.5,   # Vertical flip augmentation
            'fliplr': 0.5,   # Horizontal flip augmentation
            'mosaic': 1.0,   # Mosaic augmentation

            # Optimization
            'optimizer': 'AdamW',  # Best optimizer for transformers
            'lr0': 0.001,  # Initial learning rate
            'lrf': 0.01,   # Final learning rate
            'momentum': 0.937,
            'weight_decay': 0.0005,
            'warmup_epochs': 3,
            'warmup_momentum': 0.8,
            'warmup_bias_lr': 0.1,

            # Validation
            'val': True,
            'plots': True,
            'save': True,
            'save_period': 10,  # Save checkpoint every 10 epochs

            # Logging
            'patience': 50,  # Early stopping patience
        }

        # Start training
        print("\n🚀 Starting training...")
        print(f"⏱️  Estimated time: {self._estimate_training_time(epochs, batch_size)} hours")
        print()

        try:
            results = model.train(**training_args)

            print("\n" + "="*60)
            print("✅ TRAINING COMPLETE!")
            print("="*60)
            print(f"📊 Results: {results}")
            print(f"💾 Model saved to: {self.output_dir}")
            print()

            # Validate model
            print("🔍 Running validation...")
            metrics = model.val()
            print(f"📈 Validation mAP50: {metrics.box.map50:.3f}")
            print(f"📈 Validation mAP50-95: {metrics.box.map:.3f}")

            # Export model
            export_path = self.output_dir / 'train' / 'weights' / 'best.pt'
            if export_path.exists():
                print(f"\n✅ Best model: {export_path}")
                print("\n📝 To use this model in your app:")
                print(f"   self.food_model = YOLO('{export_path}')")

            return results

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user!")
            print(f"💾 Progress saved to: {self.output_dir}")
            return None

        except Exception as e:
            print(f"\n❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            return None

        finally:
            if use_wandb:
                wandb.finish()

    def fine_tune_pretrained(
        self,
        pretrained_model='yolov8x.pt',
        epochs=50,
        batch_size=32,
        freeze_layers=10
    ):
        """
        Fine-tune a pretrained model (faster than training from scratch).

        Args:
            pretrained_model: Path to pretrained model or model name
            epochs: Number of epochs
            batch_size: Batch size
            freeze_layers: Number of layers to freeze (speeds up training)
        """
        print(f"\n🔧 Fine-tuning pretrained model: {pretrained_model}")
        print(f"❄️  Freezing {freeze_layers} layers...")

        model = YOLO(pretrained_model)

        # Freeze backbone layers for faster training
        for i, (name, param) in enumerate(model.model.named_parameters()):
            if i < freeze_layers:
                param.requires_grad = False

        # Train with frozen layers
        results = model.train(
            data=self.data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=640,
            device=0,
            workers=16,
            project=str(self.output_dir),
            name='finetune',
            exist_ok=True,
            cache='ram',
            amp=True
        )

        return results

    def _estimate_training_time(self, epochs, batch_size):
        """Estimate training time on RTX 5090."""
        # RTX 5090: ~0.5 seconds per batch (YOLOv8x)
        # Assuming 10k images, 640x640 resolution
        estimated_batches_per_epoch = 10000 / batch_size
        estimated_seconds_per_batch = 0.5
        total_hours = (epochs * estimated_batches_per_epoch * estimated_seconds_per_batch) / 3600

        return f"{total_hours:.1f}"

    def test_model(self, model_path, test_images_dir):
        """Test trained model on sample images."""
        print(f"\n🧪 Testing model: {model_path}")

        model = YOLO(model_path)

        # Get test images
        test_images = list(Path(test_images_dir).glob('*.jpg'))
        test_images += list(Path(test_images_dir).glob('*.png'))

        if not test_images:
            print(f"❌ No test images found in {test_images_dir}")
            return

        print(f"📸 Testing on {len(test_images)} images...")

        for img_path in test_images[:5]:  # Test first 5 images
            print(f"\n📷 Testing: {img_path.name}")

            results = model(str(img_path))

            for result in results:
                boxes = result.boxes
                if boxes is not None and len(boxes) > 0:
                    print(f"   ✅ Detected {len(boxes)} objects:")
                    for box in boxes:
                        cls_id = int(box.cls)
                        conf = float(box.conf)
                        cls_name = model.names[cls_id]
                        print(f"      - {cls_name}: {conf:.2f}")
                else:
                    print("   ❌ No objects detected")

            # Save annotated image
            annotated = result.plot()
            output_path = Path('test_results') / img_path.name
            output_path.parent.mkdir(exist_ok=True)
            result.save(str(output_path))
            print(f"   💾 Saved annotated image: {output_path}")


def main():
    """Main training function."""
    import argparse

    parser = argparse.ArgumentParser(description='Train food detection model')
    parser.add_argument('--data', type=str, default='data/vision_training/food_dataset.yaml',
                        help='Path to dataset YAML file')
    parser.add_argument('--model', type=str, default='x', choices=['n', 's', 'm', 'l', 'x'],
                        help='Model size (n=nano, s=small, m=medium, l=large, x=extra large)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=32,
                        help='Batch size (RTX 5090 can handle 32+)')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for training')
    parser.add_argument('--workers', type=int, default=16,
                        help='Number of data loading workers')
    parser.add_argument('--output', type=str, default='models/food_detector',
                        help='Output directory')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from last checkpoint')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable W&B logging')
    parser.add_argument('--fine-tune', type=str,
                        help='Fine-tune from pretrained model')
    parser.add_argument('--test', type=str,
                        help='Test model on directory of images')

    args = parser.parse_args()

    # Initialize trainer
    trainer = FoodDetectorTrainer(
        model_size=args.model,
        data_yaml=args.data,
        output_dir=args.output
    )

    # Check if testing
    if args.test:
        if not Path(args.output).exists():
            print(f"❌ Model not found: {args.output}")
            sys.exit(1)

        model_path = Path(args.output) / 'train' / 'weights' / 'best.pt'
        trainer.test_model(model_path, args.test)
        return

    # Train or fine-tune
    if args.fine_tune:
        trainer.fine_tune_pretrained(
            pretrained_model=args.fine_tune,
            epochs=args.epochs,
            batch_size=args.batch
        )
    else:
        trainer.train_from_scratch(
            epochs=args.epochs,
            batch_size=args.batch,
            image_size=args.img_size,
            workers=args.workers,
            use_wandb=not args.no_wandb,
            resume=args.resume
        )


if __name__ == "__main__":
    main()
