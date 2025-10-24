# ChefGenius TensorRT API - Windows Setup

Ultra-fast recipe generation API using TensorRT on Windows with NVIDIA GPUs.

## 🚀 Quick Start (Windows)

### Prerequisites

1. **NVIDIA GPU** with CUDA Compute Capability 6.0+
2. **CUDA Toolkit 12.0+** - [Download](https://developer.nvidia.com/cuda-downloads)
3. **Python 3.8+** - [Download](https://python.org/downloads)
4. **Go 1.21+** - [Download](https://golang.org/dl/)
5. **Git for Windows** - [Download](https://git-scm.com/download/win)

### Installation

```cmd
# Clone repository
git clone <repo-url>
cd chef-genius

# Option 1: One-click build (recommended)
build.bat

# Option 2: Manual steps
convert.bat          # Convert model to TensorRT
cd api-server
go mod download
go build -o chef-genius-api.exe .
chef-genius-api.exe
```

## 📋 Detailed Setup

### 1. Environment Setup

**Install CUDA Toolkit:**
```cmd
# Download and install CUDA 12.0+ from NVIDIA
# Add to PATH: C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin
# Verify installation:
nvcc --version
nvidia-smi
```

**Install Python:**
```cmd
# Download Python 3.8+ from python.org
# Make sure to check "Add Python to PATH" during installation
# Verify:
python --version
pip --version
```

**Install Go:**
```cmd
# Download Go 1.21+ from golang.org
# Add to PATH (usually automatic)
# Verify:
go version
```

### 2. Model Conversion

**Automatic Conversion (Recommended):**
```cmd
# Run the conversion script
convert.bat
```

**Manual Conversion:**
```cmd
# Create virtual environment
python -m venv venv-conversion
venv-conversion\Scripts\activate

# Install dependencies
pip install -r requirements-conversion.txt
pip install nvidia-tensorrt

# Run conversion
python convert_to_tensorrt.py --model-path .\models\recipe-gen --output-dir .\models\tensorrt --precision fp16
```

**PowerShell Alternative:**
```powershell
# Run PowerShell as Administrator
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\convert.ps1
```

### 3. API Server Build

**Using Makefile (if you have make):**
```cmd
# Install make for Windows: https://www.gnu.org/software/make/
cd api-server
make -f Makefile.windows build
make -f Makefile.windows run
```

**Manual Build:**
```cmd
cd api-server
go mod download
set CGO_ENABLED=1
go build -o chef-genius-api.exe .
chef-genius-api.exe
```

## 🔧 Configuration

### Environment Variables

Create `.env` file in `api-server` directory:
```env
PORT=8080
CUDA_VISIBLE_DEVICES=0
MODEL_PATH=../models/tensorrt/model.trt
POOL_SIZE=4
MAX_BATCH_SIZE=10
LOG_LEVEL=info
```

### TensorRT Settings

Edit `api-server/pool.go`:
```go
// Adjust pool size based on your GPU memory
maxEngines := 4  // Reduce if you have limited VRAM

// Model path (automatically set by conversion script)
modelPath := "../models/tensorrt/model.trt"
```

## 🧪 Testing

### Health Check
```cmd
curl http://localhost:8080/health
```

### Generate Recipe
```cmd
curl -X POST http://localhost:8080/api/v1/generate ^
  -H "Content-Type: application/json" ^
  -d "{\"ingredients\":[\"chicken\",\"rice\",\"vegetables\"],\"max_tokens\":512}"
```

### Benchmark
```cmd
curl -X POST http://localhost:8080/api/v1/benchmark?iterations=100
```

## 🐛 Troubleshooting

### Common Issues

**1. CUDA Not Found**
```cmd
# Check CUDA installation
nvcc --version
# If not found, add to PATH:
set PATH=%PATH%;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin
```

**2. TensorRT Installation Failed**
```cmd
# Try different installation methods:
pip install nvidia-tensorrt
# Or download manually from NVIDIA Developer website
```

**3. Go Build Errors**
```cmd
# Ensure CGO is enabled
set CGO_ENABLED=1
# Check Go version
go version
# Should be 1.21 or higher
```

**4. Out of GPU Memory**
```cmd
# Reduce engine pool size in pool.go
maxEngines := 2  # Instead of 4

# Or use smaller batch sizes
MAX_BATCH_SIZE=5
```

**5. Python Virtual Environment Issues**
```cmd
# Delete and recreate venv
rmdir /s venv-conversion
python -m venv venv-conversion
venv-conversion\Scripts\activate
pip install -r requirements-conversion.txt
```

### Performance Optimization

**GPU Memory Optimization:**
```cmd
# Monitor GPU usage
nvidia-smi -l 1

# Reduce model precision (in convert script)
python convert_to_tensorrt.py --precision int8  # Instead of fp16
```

**CPU Optimization:**
```cmd
# Enable Go prefork in main.go
Prefork: true  // Multi-process mode
```

## 📊 Performance Expectations

### Typical Results on RTX 4090:
- **Conversion Time:** 5-15 minutes
- **Inference Latency:** 5-20ms per request
- **Throughput:** 1000+ requests/second
- **Memory Usage:** 2-8GB VRAM (depending on model size)

### Optimization Tips:
1. Use FP16 precision for best speed/quality balance
2. Increase engine pool size if you have more VRAM
3. Use batch requests for maximum throughput
4. Enable GPU monitoring to optimize settings

## 🔄 Updates

### Update Model:
```cmd
# Replace model files in .\models\recipe-gen\
# Re-run conversion
convert.bat
```

### Update API:
```cmd
cd api-server
go mod download
go build -o chef-genius-api.exe .
```

## 📝 Logs

**View Logs:**
```cmd
# API logs are printed to console
# Or redirect to file:
chef-genius-api.exe > api.log 2>&1
```

**TensorRT Logs:**
```cmd
# Check conversion logs in conversion output
# GPU logs via nvidia-smi
```

## 🚀 Production Deployment

### Windows Service

Create `chef-genius-service.bat`:
```cmd
@echo off
cd /d "C:\path\to\chef-genius\api-server"
chef-genius-api.exe
```

### Docker (Windows Containers)

```dockerfile
# Use Windows base image with CUDA support
FROM mcr.microsoft.com/windows/servercore:ltsc2022
# Add CUDA and TensorRT installation steps
# Copy application files
```

### IIS Reverse Proxy

Configure IIS to proxy requests to localhost:8080.

## 📞 Support

For Windows-specific issues:
1. Check Windows Event Viewer for system errors
2. Verify all PATH variables are set correctly
3. Run Command Prompt as Administrator if needed
4. Check Windows Defender/Antivirus exclusions

Common Windows paths:
- CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0`
- Python: `C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python3x`
- Go: `C:\Program Files\Go`