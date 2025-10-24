@echo off
REM Install TensorRT on Windows - NVIDIA Package Index Method

echo ========================================
echo 🔧 Installing TensorRT for Windows
echo ========================================

REM Activate virtual environment
if exist "venv-conversion\Scripts\activate.bat" (
    call venv-conversion\Scripts\activate.bat
) else if exist ".venv\Scripts\activate.bat" (
    call .venv\Scripts\activate.bat
) else (
    echo ❌ Virtual environment not found
    echo Please run convert.bat first to create the environment
    pause
    exit /b 1
)

echo ✅ Virtual environment activated

REM Method 1: NVIDIA Package Index (Recommended)
echo.
echo 📦 Installing via NVIDIA Package Index...
pip install nvidia-pyindex
if %errorlevel% neq 0 (
    echo ⚠️  NVIDIA pyindex installation failed, trying alternative methods...
    goto :method2
)

echo ✅ NVIDIA pyindex installed successfully
pip install nvidia-tensorrt
if %errorlevel% neq 0 (
    echo ⚠️  TensorRT installation via pyindex failed, trying alternative methods...
    goto :method2
) else (
    echo ✅ TensorRT installed successfully via NVIDIA Package Index!
    goto :test_install
)

:method2
echo.
echo 📦 Method 2: Installing specific CUDA version...
REM Try CUDA 12.x compatible version
pip install --extra-index-url https://pypi.nvidia.com nvidia-tensorrt==8.6.1.post1
if %errorlevel% neq 0 (
    echo ⚠️  Specific version failed, trying method 3...
    goto :method3
) else (
    echo ✅ TensorRT installed successfully!
    goto :test_install
)

:method3
echo.
echo 📦 Method 3: Installing via conda-forge...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo ⚠️  Conda not found, trying method 4...
    goto :method4
)

conda install -c conda-forge tensorrt -y
if %errorlevel% neq 0 (
    echo ⚠️  Conda installation failed, trying method 4...
    goto :method4
) else (
    echo ✅ TensorRT installed via conda!
    goto :test_install
)

:method4
echo.
echo 📦 Method 4: Manual installation guide...
echo ❌ All automatic installation methods failed.
echo.
echo Please install TensorRT manually:
echo.
echo Option A - NVIDIA Developer (Recommended):
echo 1. Go to: https://developer.nvidia.com/tensorrt
echo 2. Download TensorRT 8.6.1 for CUDA 12.x
echo 3. Extract and follow installation guide
echo 4. Add to Python path
echo.
echo Option B - Docker:
echo 1. Use NVIDIA TensorRT container
echo 2. docker pull nvcr.io/nvidia/tensorrt:23.12-py3
echo.
echo Option C - WSL2:
echo 1. Install WSL2 with Ubuntu
echo 2. Install CUDA in WSL2
echo 3. Use Linux installation method
echo.
pause
exit /b 1

:test_install
echo.
echo 🧪 Testing TensorRT installation...
python -c "import tensorrt; print('✅ TensorRT version:', tensorrt.__version__); print('✅ TensorRT installation successful!')"
if %errorlevel% neq 0 (
    echo ❌ TensorRT test failed
    echo.
    echo The installation may be incomplete. Please try manual installation:
    echo https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html
    pause
    exit /b 1
)

echo.
echo 🎉 TensorRT installation completed successfully!
echo You can now run the model conversion:
echo.
echo python convert_to_tensorrt.py --model-path .\models\recipe-gen --output-dir .\models\tensorrt --precision fp16
echo.
pause