#!/bin/bash
# Quick Start: Train Vision Model for Fridge-to-Recipe App
# Complete automated setup for RTX 5090

set -e  # Exit on error

echo "🍳 ChefGenius Vision Model Quick Start"
echo "======================================"
echo ""
echo "This script will:"
echo "  1. Install required dependencies"
echo "  2. Download Food-101 dataset (5GB)"
echo "  3. Convert to YOLO format"
echo "  4. Train YOLOv8 model on your RTX 5090"
echo "  5. Test the trained model"
echo ""
echo "⏱️  Estimated time: 8-12 hours (mostly training)"
echo "💾 Disk space needed: 20GB"
echo ""

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

# Step 1: Check prerequisites
echo ""
echo "📋 Step 1: Checking prerequisites..."
echo "===================================="

# Check CUDA
if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "❌ CUDA not available!"
    echo "   Please install CUDA 12.x first"
    exit 1
fi

GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
echo "✅ GPU detected: $GPU_NAME"

# Check disk space
AVAILABLE_SPACE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
if [ "$AVAILABLE_SPACE" -lt 20 ]; then
    echo "❌ Not enough disk space!"
    echo "   Available: ${AVAILABLE_SPACE}GB"
    echo "   Required: 20GB"
    exit 1
fi

echo "✅ Disk space: ${AVAILABLE_SPACE}GB available"

# Step 2: Install dependencies
echo ""
echo "📦 Step 2: Installing dependencies..."
echo "===================================="

pip install -q ultralytics opencv-python Pillow pyyaml tqdm wandb requests || {
    echo "❌ Failed to install dependencies"
    exit 1
}

echo "✅ Dependencies installed"

# Step 3: Download Food-101 dataset
echo ""
echo "📥 Step 3: Downloading Food-101 dataset (5GB)..."
echo "================================================"
echo "⏱️  This will take 30-60 minutes depending on your internet speed"
echo ""

if [ -d "data/vision_training/food-101" ]; then
    echo "✅ Food-101 already downloaded"
else
    python3 cli/setup_vision_training.py --auto-download food101 || {
        echo "❌ Failed to download Food-101"
        echo "💡 You can download manually from:"
        echo "   http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
        exit 1
    }
    echo "✅ Food-101 downloaded"
fi

# Step 4: Convert to YOLO format
echo ""
echo "🔄 Step 4: Converting to YOLO format..."
echo "========================================"

if [ -f "data/vision_training/food101_yolo/data.yaml" ]; then
    echo "✅ Dataset already converted"
else
    python3 cli/convert_food101_to_yolo.py \
        --input data/vision_training/food-101 \
        --output data/vision_training/food101_yolo \
        --split 0.8 || {
        echo "❌ Failed to convert dataset"
        exit 1
    }
    echo "✅ Dataset converted to YOLO format"
fi

# Step 5: Train model
echo ""
echo "🚀 Step 5: Training YOLOv8 model on RTX 5090..."
echo "================================================"
echo ""
echo "⏱️  Estimated time: 6-12 hours"
echo "🎯 Target mAP: 75-85%"
echo ""
echo "Training configuration:"
echo "  Model: YOLOv8x (extra large)"
echo "  Epochs: 100"
echo "  Batch size: 32"
echo "  Image size: 640x640"
echo "  Workers: 16"
echo ""

read -p "Start training now? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "⏸️  Training skipped"
    echo ""
    echo "To train later, run:"
    echo "  python cli/train_food_detector.py --data data/vision_training/food101_yolo/data.yaml"
    exit 0
fi

# Check if training already completed
if [ -f "models/food_detector/train/weights/best.pt" ]; then
    echo "✅ Model already trained!"
    echo "💾 Model location: models/food_detector/train/weights/best.pt"
else
    python3 cli/train_food_detector.py \
        --data data/vision_training/food101_yolo/data.yaml \
        --model x \
        --epochs 100 \
        --batch 32 \
        --workers 16 \
        --output models/food_detector || {
        echo "❌ Training failed"
        echo "📋 Check logs in models/food_detector/train/"
        exit 1
    }

    echo ""
    echo "✅ Training complete!"
    echo "💾 Model saved to: models/food_detector/train/weights/best.pt"
fi

# Step 6: Test model
echo ""
echo "🧪 Step 6: Testing trained model..."
echo "===================================="

# Create test directory if needed
mkdir -p test_images

if [ "$(ls -A test_images/*.jpg 2>/dev/null)" ]; then
    echo "📸 Testing on images in test_images/"

    python3 cli/train_food_detector.py \
        --test test_images/ \
        --output models/food_detector

    echo ""
    echo "✅ Test complete!"
    echo "📁 Results saved to: test_results/"
else
    echo "💡 No test images found in test_images/"
    echo "   Add some fridge photos to test_images/ and run:"
    echo "   python cli/train_food_detector.py --test test_images/"
fi

# Final summary
echo ""
echo "=" | tr '=' '='
echo "🎉 SETUP COMPLETE!"
echo "=" | tr '=' '='
echo ""
echo "✅ Vision model trained and ready to use!"
echo ""
echo "📊 Model Information:"
python3 -c "
from ultralytics import YOLO
model = YOLO('models/food_detector/train/weights/best.pt')
print(f'   Classes: {len(model.names)}')
print(f'   Sample classes: {list(model.names.values())[:10]}...')
" 2>/dev/null || echo "   (Run test to see model info)"

echo ""
echo "📝 Next Steps:"
echo ""
echo "1. Update backend to use trained model:"
echo "   Edit: backend/app/services/vision_service.py"
echo "   Change: self.food_model = YOLO('models/food_detector/train/weights/best.pt')"
echo ""
echo "2. Test API endpoint:"
echo "   python backend/test_vision_api.py"
echo ""
echo "3. Connect mobile app:"
echo "   Update API URL in mobile_demo/lib/services/api_service.dart"
echo ""
echo "4. Deploy to production:"
echo "   docker build -f Dockerfile.vision -t chefgenius-vision ."
echo "   docker run --gpus all -p 8000:8000 chefgenius-vision"
echo ""
echo "🎯 Your fridge-to-recipe system is ready!"
echo ""
