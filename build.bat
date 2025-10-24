@echo off
REM Quick build script for Windows

setlocal EnableDelayedExpansion

echo ========================================
echo 🔨 ChefGenius TensorRT API Build
echo ========================================

REM Check if conversion was done
if not exist "models\tensorrt\model.trt" (
    echo ⚠️  TensorRT model not found. Running conversion first...
    call convert.bat
    if !errorlevel! neq 0 (
        echo ❌ Conversion failed
        pause
        exit /b 1
    )
)

REM Navigate to API server directory
cd api-server

REM Check Go installation
go version >nul 2>&1
if !errorlevel! neq 0 (
    echo ❌ Go not found. Please install Go 1.21+ and add to PATH.
    echo Download from: https://golang.org/dl/
    pause
    exit /b 1
)

echo ✅ Go found
echo.

REM Install dependencies
echo 📦 Installing Go dependencies...
go mod download
go mod tidy

REM Create bin directory
if not exist "bin" mkdir bin

REM Build the application
echo 🔨 Building TensorRT API...
set CGO_ENABLED=1
go build -o bin\chef-genius-api.exe .

if !errorlevel! neq 0 (
    echo ❌ Build failed
    pause
    exit /b 1
)

echo ✅ Build successful!
echo 📁 Binary: api-server\bin\chef-genius-api.exe
echo.

REM Check if we should run
set /p "runChoice=🚀 Run the API server now? (y/N): "
if /i "!runChoice!"=="y" (
    echo.
    echo 🚀 Starting ChefGenius TensorRT API...
    echo Server will start on http://localhost:8080
    echo Press Ctrl+C to stop
    echo.
    bin\chef-genius-api.exe
) else (
    echo.
    echo To run the API server manually:
    echo   cd api-server
    echo   bin\chef-genius-api.exe
    echo.
)

pause