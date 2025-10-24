#!/bin/bash
# Convert Transformers model to TensorRT for Go API

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 ChefGenius Model Conversion Pipeline${NC}"
echo "Converting Transformers model to TensorRT for ultra-fast inference..."

# Configuration
MODEL_PATH="./models/recipe-gen"
OUTPUT_DIR="./models/tensorrt"
PRECISION="fp16"
VENV_PATH="./venv-conversion"

# Check if model exists
if [ ! -d "$MODEL_PATH" ]; then
    echo -e "${RED}❌ Model directory not found: $MODEL_PATH${NC}"
    exit 1
fi

echo -e "${YELLOW}📁 Model path: $MODEL_PATH${NC}"
echo -e "${YELLOW}📁 Output directory: $OUTPUT_DIR${NC}"
echo -e "${YELLOW}⚡ Precision: $PRECISION${NC}"

# Check CUDA
echo -e "\n${YELLOW}🔍 Checking CUDA installation...${NC}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}❌ CUDA not found. Please install CUDA toolkit.${NC}"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//')
echo -e "${GREEN}✅ CUDA version: $CUDA_VERSION${NC}"

# Check GPU
if ! nvidia-smi &> /dev/null; then
    echo -e "${RED}❌ No NVIDIA GPU found or driver not installed.${NC}"
    exit 1
fi

GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -1)
echo -e "${GREEN}✅ GPU: $GPU_INFO${NC}"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_PATH" ]; then
    echo -e "\n${YELLOW}🐍 Creating Python virtual environment...${NC}"
    python3 -m venv "$VENV_PATH"
fi

# Activate virtual environment
source "$VENV_PATH/bin/activate"

# Install requirements
echo -e "\n${YELLOW}📦 Installing Python dependencies...${NC}"
pip install --upgrade pip
pip install -r requirements-conversion.txt

# Install TensorRT (user needs to do this manually or via conda)
echo -e "\n${YELLOW}🔧 Checking TensorRT installation...${NC}"
if ! python -c "import tensorrt" &> /dev/null; then
    echo -e "${YELLOW}⚠️  TensorRT not found. Installing via pip...${NC}"
    echo -e "${YELLOW}Note: You may need to install TensorRT manually for your CUDA version.${NC}"
    
    # Try to install TensorRT
    pip install nvidia-tensorrt || {
        echo -e "${RED}❌ Failed to install TensorRT automatically.${NC}"
        echo -e "${YELLOW}Please install TensorRT manually:${NC}"
        echo "1. Download from: https://developer.nvidia.com/tensorrt"
        echo "2. Or use conda: conda install -c conda-forge tensorrt"
        echo "3. Or pip: pip install nvidia-tensorrt"
        exit 1
    }
fi

# Run conversion
echo -e "\n${GREEN}🔄 Starting model conversion...${NC}"
python convert_to_tensorrt.py \
    --model-path "$MODEL_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --precision "$PRECISION"

# Check if conversion was successful
if [ -f "$OUTPUT_DIR/model.trt" ]; then
    echo -e "\n${GREEN}✅ Conversion completed successfully!${NC}"
    echo -e "${GREEN}📁 TensorRT engine: $OUTPUT_DIR/model.trt${NC}"
    
    # Show file sizes
    echo -e "\n${YELLOW}📊 File sizes:${NC}"
    ls -lh "$OUTPUT_DIR"/model.* 2>/dev/null || true
    
    # Update Go API configuration
    echo -e "\n${YELLOW}🔧 Updating Go API configuration...${NC}"
    sed -i.bak "s|modelPath := \".*\"|modelPath := \"$OUTPUT_DIR/model.trt\"|" api-server/pool.go
    echo -e "${GREEN}✅ Go API updated to use TensorRT engine${NC}"
    
    # Test the conversion
    echo -e "\n${YELLOW}🧪 Testing converted model...${NC}"
    python convert_to_tensorrt.py \
        --model-path "$MODEL_PATH" \
        --output-dir "$OUTPUT_DIR" \
        --test-only
    
    echo -e "\n${GREEN}🎉 Model conversion pipeline completed!${NC}"
    echo -e "${GREEN}You can now build and run the Go API with ultra-fast TensorRT inference.${NC}"
    echo -e "${YELLOW}Next steps:${NC}"
    echo "1. cd api-server"
    echo "2. make build"
    echo "3. make run"
    
else
    echo -e "\n${RED}❌ Conversion failed. Check the logs above.${NC}"
    exit 1
fi

# Deactivate virtual environment
deactivate