@echo off
REM Convert Transformers model to TensorRT for Go API on Windows

setlocal EnableDelayedExpansion

echo ========================================
echo 🚀 ChefGenius Model Conversion Pipeline
echo ========================================
echo Converting Transformers model to TensorRT for ultra-fast inference...
echo.

REM Configuration
set MODEL_PATH=.\models\recipe-gen
set OUTPUT_DIR=.\models\tensorrt
set PRECISION=fp16
set VENV_PATH=.\venv-conversion

echo 📁 Model path: %MODEL_PATH%
echo 📁 Output directory: %OUTPUT_DIR%
echo ⚡ Precision: %PRECISION%
echo.

REM Check if model exists
if not exist "%MODEL_PATH%" (
    echo ❌ Model directory not found: %MODEL_PATH%
    pause
    exit /b 1
)

REM Check CUDA
echo 🔍 Checking CUDA installation...
nvcc --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ CUDA not found. Please install CUDA toolkit.
    echo Download from: https://developer.nvidia.com/cuda-downloads
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('nvcc --version ^| findstr "release"') do (
    echo ✅ CUDA found: %%i
)

REM Check GPU
nvidia-smi >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ No NVIDIA GPU found or driver not installed.
    echo Please install NVIDIA GPU drivers.
    pause
    exit /b 1
)

echo ✅ GPU check passed
echo.

REM Check Python
python --version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Python not found. Please install Python 3.8+ and add to PATH.
    echo Download from: https://python.org/downloads
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment if it doesn't exist
if not exist "%VENV_PATH%" (
    echo 🐍 Creating Python virtual environment...
    python -m venv "%VENV_PATH%"
    if !errorlevel! neq 0 (
        echo ❌ Failed to create virtual environment
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call "%VENV_PATH%\Scripts\activate.bat"

REM Upgrade pip
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing Python dependencies...
pip install -r requirements-conversion.txt
if !errorlevel! neq 0 (
    echo ❌ Failed to install dependencies
    pause
    exit /b 1
)

REM Check TensorRT installation
echo 🔧 Checking TensorRT installation...
python -c "import tensorrt" >nul 2>&1
if !errorlevel! neq 0 (
    echo ⚠️  TensorRT not found. Installing via pip...
    echo Note: You may need to install TensorRT manually for your CUDA version.
    
    REM Try to install TensorRT
    pip install nvidia-tensorrt
    if !errorlevel! neq 0 (
        echo ❌ Failed to install TensorRT automatically.
        echo.
        echo Please install TensorRT manually:
        echo 1. Download from: https://developer.nvidia.com/tensorrt
        echo 2. Or use conda: conda install -c conda-forge tensorrt
        echo 3. Or pip: pip install nvidia-tensorrt
        pause
        exit /b 1
    )
)

echo ✅ TensorRT check passed
echo.

REM Run conversion
echo 🔄 Starting model conversion...
echo This may take several minutes depending on your GPU...
echo.

python convert_to_tensorrt.py --model-path "%MODEL_PATH%" --output-dir "%OUTPUT_DIR%" --precision "%PRECISION%"

if !errorlevel! neq 0 (
    echo ❌ Conversion failed. Check the logs above.
    pause
    exit /b 1
)

REM Check if conversion was successful
if exist "%OUTPUT_DIR%\model.trt" (
    echo.
    echo ✅ Conversion completed successfully!
    echo 📁 TensorRT engine: %OUTPUT_DIR%\model.trt
    echo.
    
    REM Show file sizes
    echo 📊 File sizes:
    dir "%OUTPUT_DIR%\model.*" 2>nul
    echo.
    
    REM Update Go API configuration
    echo 🔧 Updating Go API configuration...
    powershell -Command "(Get-Content api-server\pool.go) -replace 'modelPath := \".*\"', 'modelPath := \"../models/tensorrt/model.trt\"' | Set-Content api-server\pool.go"
    echo ✅ Go API updated to use TensorRT engine
    echo.
    
    REM Test the conversion
    echo 🧪 Testing converted model...
    python convert_to_tensorrt.py --model-path "%MODEL_PATH%" --output-dir "%OUTPUT_DIR%" --test-only
    
    echo.
    echo 🎉 Model conversion pipeline completed!
    echo ✅ You can now build and run the Go API with ultra-fast TensorRT inference.
    echo.
    echo Next steps:
    echo 1. cd api-server
    echo 2. go mod download
    echo 3. go build -o chef-genius-api.exe .
    echo 4. chef-genius-api.exe
    echo.
    
) else (
    echo.
    echo ❌ Conversion failed. TensorRT engine not found.
    echo Check the error messages above.
    pause
    exit /b 1
)

REM Deactivate virtual environment
call "%VENV_PATH%\Scripts\deactivate.bat"

echo Press any key to exit...
pause >nul