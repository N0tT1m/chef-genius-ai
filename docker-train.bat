@echo off
REM Chef Genius Docker Training Script for Windows

echo 🚀 Starting Chef Genius Training with Docker + PyTorch Compilation

REM Build the training container
docker-compose -f docker-compose.training.yml build

REM Run training with optimized settings (resuming from checkpoint-1000)
docker-compose -f docker-compose.training.yml run --rm chef-genius-training ^
  python cli/complete_optimized_training.py ^
  --epochs 5 ^
  --batch-size 16 ^
  --gradient-accumulation-steps 2 ^
  --enable-mixed-precision ^
  --enable-profiling ^
  --profile-schedule "wait=2;warmup=2;active=5;repeat=3" ^
  --model-output /workspace/models/recipe_generation_flan-t5-large ^
  --pretrained-model google/flan-t5-large ^
  --alert-phone "+18125841533" ^
  --discord-webhook "https://discord.com/api/webhooks/1386109570283343953/uGkhj9dpuCg09SbKzZ0Tx2evugJrchQv-nrq3w0r_xi3w8si-XBpQJuxq_p_bcQlhB9W" ^
  --resume-from-checkpoint /workspace/models/recipe_generation_flan-t5-large/checkpoint-1000

echo ✅ Training completed!
pause