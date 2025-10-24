@echo off
REM Chef Genius Checkpoint Verification Script for Windows

echo 🔍 Verifying checkpoint before Docker training...

set CHECKPOINT_DIR=.\models\recipe_generation_flan-t5-large\checkpoint-1000

if exist "%CHECKPOINT_DIR%" (
    echo ✅ Checkpoint directory found: %CHECKPOINT_DIR%
    
    REM Check for essential files
    if exist "%CHECKPOINT_DIR%\config.json" (
        echo ✅ config.json found
    ) else (
        echo ❌ config.json missing
        exit /b 1
    )
    
    if exist "%CHECKPOINT_DIR%\pytorch_model.bin" (
        echo ✅ pytorch_model.bin found
    ) else if exist "%CHECKPOINT_DIR%\model.safetensors" (
        echo ✅ model.safetensors found
    ) else (
        echo ❌ Model weights missing (pytorch_model.bin or model.safetensors)
        exit /b 1
    )
    
    if exist "%CHECKPOINT_DIR%\tokenizer.json" (
        echo ✅ tokenizer.json found
    ) else (
        echo ❌ tokenizer.json missing
        exit /b 1
    )
    
    echo.
    echo ✅ Checkpoint verification passed!
    echo 🚀 Ready to start Docker training from step 1,000
    echo.
    echo Run: docker-train.bat to start training
    
) else (
    echo ❌ Checkpoint directory not found: %CHECKPOINT_DIR%
    echo.
    echo Expected structure:
    echo %CHECKPOINT_DIR%\
    echo ├── config.json
    echo ├── pytorch_model.bin (or model.safetensors)
    echo ├── tokenizer.json
    echo ├── tokenizer_config.json
    echo └── training_args.bin
    echo.
    echo Please ensure your checkpoint exists before running Docker training.
    exit /b 1
)

pause