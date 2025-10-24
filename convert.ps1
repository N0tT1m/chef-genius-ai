# PowerShell script for TensorRT conversion on Windows
# ChefGenius Model Conversion Pipeline

param(
    [string]$ModelPath = ".\models\recipe-gen",
    [string]$OutputDir = ".\models\tensorrt", 
    [string]$Precision = "fp16",
    [switch]$TestOnly = $false
)

# Set console colors
function Write-ColorOutput {
    param(
        [string]$Message,
        [string]$Color = "White"
    )
    Write-Host $Message -ForegroundColor $Color
}

function Write-Success { param([string]$Message) Write-ColorOutput "✅ $Message" "Green" }
function Write-Error { param([string]$Message) Write-ColorOutput "❌ $Message" "Red" }
function Write-Warning { param([string]$Message) Write-ColorOutput "⚠️ $Message" "Yellow" }
function Write-Info { param([string]$Message) Write-ColorOutput "ℹ️ $Message" "Cyan" }

# Header
Write-ColorOutput "========================================" "Magenta"
Write-ColorOutput "🚀 ChefGenius Model Conversion Pipeline" "Magenta"
Write-ColorOutput "========================================" "Magenta"
Write-ColorOutput "Converting Transformers model to TensorRT for ultra-fast inference..."
Write-Host ""

# Configuration
$VenvPath = ".\venv-conversion"

Write-Info "📁 Model path: $ModelPath"
Write-Info "📁 Output directory: $OutputDir"
Write-Info "⚡ Precision: $Precision"
Write-Info "🐍 Virtual env: $VenvPath"
Write-Host ""

# Validate inputs
if (-not (Test-Path $ModelPath)) {
    Write-Error "Model directory not found: $ModelPath"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check CUDA
Write-Info "🔍 Checking CUDA installation..."
try {
    $cudaVersion = & nvcc --version 2>$null | Select-String "release" | ForEach-Object { $_.ToString() }
    if ($cudaVersion) {
        Write-Success "CUDA found: $cudaVersion"
    } else {
        throw "NVCC not found"
    }
} catch {
    Write-Error "CUDA not found. Please install CUDA toolkit."
    Write-Warning "Download from: https://developer.nvidia.com/cuda-downloads"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check GPU
Write-Info "🔍 Checking NVIDIA GPU..."
try {
    $gpuInfo = & nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>$null | Select-Object -First 1
    if ($gpuInfo) {
        Write-Success "GPU found: $gpuInfo"
    } else {
        throw "nvidia-smi failed"
    }
} catch {
    Write-Error "No NVIDIA GPU found or driver not installed."
    Write-Warning "Please install NVIDIA GPU drivers."
    Read-Host "Press Enter to exit"
    exit 1
}

# Check Python
Write-Info "🔍 Checking Python installation..."
try {
    $pythonVersion = & python --version 2>$null
    if ($pythonVersion) {
        Write-Success "Python found: $pythonVersion"
    } else {
        throw "Python not found"
    }
} catch {
    Write-Error "Python not found. Please install Python 3.8+ and add to PATH."
    Write-Warning "Download from: https://python.org/downloads"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""

# Create virtual environment
if (-not (Test-Path $VenvPath)) {
    Write-Info "🐍 Creating Python virtual environment..."
    try {
        & python -m venv $VenvPath
        Write-Success "Virtual environment created"
    } catch {
        Write-Error "Failed to create virtual environment"
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-Info "🐍 Using existing virtual environment"
}

# Activate virtual environment
Write-Info "🔧 Activating virtual environment..."
$activateScript = Join-Path $VenvPath "Scripts\Activate.ps1"

if (Test-Path $activateScript) {
    # PowerShell activation
    & $activateScript
} else {
    # Fallback to batch activation
    $activateBat = Join-Path $VenvPath "Scripts\activate.bat"
    if (Test-Path $activateBat) {
        cmd /c $activateBat
    } else {
        Write-Error "Virtual environment activation script not found"
        exit 1
    }
}

# Upgrade pip
Write-Info "📦 Upgrading pip..."
try {
    & python -m pip install --upgrade pip --quiet
    Write-Success "Pip upgraded"
} catch {
    Write-Warning "Failed to upgrade pip, continuing..."
}

# Install requirements
Write-Info "📦 Installing Python dependencies..."
Write-Warning "This may take several minutes..."

try {
    & pip install -r requirements-conversion.txt
    Write-Success "Dependencies installed"
} catch {
    Write-Error "Failed to install dependencies"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check TensorRT
Write-Info "🔧 Checking TensorRT installation..."
try {
    & python -c "import tensorrt; print('TensorRT version:', tensorrt.__version__)" 2>$null
    Write-Success "TensorRT found"
} catch {
    Write-Warning "TensorRT not found. Installing via pip..."
    Write-Info "Note: You may need to install TensorRT manually for your CUDA version."
    
    try {
        & pip install nvidia-tensorrt
        Write-Success "TensorRT installed via pip"
    } catch {
        Write-Error "Failed to install TensorRT automatically."
        Write-Warning ""
        Write-Warning "Please install TensorRT manually:"
        Write-Warning "1. Download from: https://developer.nvidia.com/tensorrt"
        Write-Warning "2. Or use conda: conda install -c conda-forge tensorrt"
        Write-Warning "3. Or pip: pip install nvidia-tensorrt"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

Write-Host ""

# Run conversion or test
if ($TestOnly) {
    Write-Info "🧪 Testing existing models..."
    try {
        & python convert_to_tensorrt.py --model-path $ModelPath --output-dir $OutputDir --test-only
        Write-Success "Model testing completed"
    } catch {
        Write-Error "Model testing failed"
        exit 1
    }
} else {
    Write-Info "🔄 Starting model conversion..."
    Write-Warning "This may take several minutes depending on your GPU..."
    Write-Host ""
    
    try {
        & python convert_to_tensorrt.py --model-path $ModelPath --output-dir $OutputDir --precision $Precision
        
        # Check if conversion was successful
        $tensorrtFile = Join-Path $OutputDir "model.trt"
        if (Test-Path $tensorrtFile) {
            Write-Host ""
            Write-Success "Conversion completed successfully!"
            Write-Success "📁 TensorRT engine: $tensorrtFile"
            Write-Host ""
            
            # Show file sizes
            Write-Info "📊 File sizes:"
            Get-ChildItem "$OutputDir\model.*" | ForEach-Object {
                $sizeKB = [math]::Round($_.Length / 1KB, 2)
                $sizeMB = [math]::Round($_.Length / 1MB, 2)
                Write-Host "  $($_.Name): $sizeMB MB ($sizeKB KB)"
            }
            Write-Host ""
            
            # Update Go API configuration
            Write-Info "🔧 Updating Go API configuration..."
            try {
                $poolGoPath = "api-server\pool.go"
                if (Test-Path $poolGoPath) {
                    $content = Get-Content $poolGoPath -Raw
                    $newContent = $content -replace 'modelPath := ".*"', 'modelPath := "../models/tensorrt/model.trt"'
                    Set-Content $poolGoPath -Value $newContent
                    Write-Success "Go API updated to use TensorRT engine"
                } else {
                    Write-Warning "Go API pool.go not found, manual update required"
                }
            } catch {
                Write-Warning "Failed to update Go API automatically"
            }
            Write-Host ""
            
            # Test the conversion
            Write-Info "🧪 Testing converted model..."
            try {
                & python convert_to_tensorrt.py --model-path $ModelPath --output-dir $OutputDir --test-only
            } catch {
                Write-Warning "Model testing failed, but conversion completed"
            }
            
            Write-Host ""
            Write-ColorOutput "🎉 Model conversion pipeline completed!" "Green"
            Write-Success "You can now build and run the Go API with ultra-fast TensorRT inference."
            Write-Host ""
            Write-Info "Next steps:"
            Write-Host "1. cd api-server"
            Write-Host "2. go mod download"
            Write-Host "3. go build -o chef-genius-api.exe ."
            Write-Host "4. .\chef-genius-api.exe"
            Write-Host ""
            
        } else {
            Write-Error "Conversion failed. TensorRT engine not found."
            Write-Error "Check the error messages above."
            exit 1
        }
        
    } catch {
        Write-Error "Conversion failed: $($_.Exception.Message)"
        exit 1
    }
}

# Deactivate virtual environment
try {
    & deactivate 2>$null
} catch {
    # Ignore deactivation errors
}

Write-Host ""
Write-ColorOutput "Conversion pipeline finished." "Magenta"
Read-Host "Press Enter to exit"