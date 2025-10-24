#!/bin/bash

# Chef Genius Docker Training Script

set -e

echo "🚀 Starting Chef Genius Training with Docker + PyTorch Compilation"

# Load environment variables
if [ -f .env.training ]; then
    export $(cat .env.training | xargs)
fi

# Build and run training with exact parameters
docker-compose -f docker-compose.training.yml build

docker-compose -f docker-compose.training.yml run --rm chef-genius-training \
  python cli/complete_optimized_training.py \
  --epochs 5 \
  --batch-size 48 \
  --gradient-accumulation-steps 1 \
  --enable-mixed-precision \
  --dataloader-num-workers 16 \
  --disable-compilation \
  --disable-cudagraphs \
  --model-output /workspace/models/recipe_generation_flan-t5-large \
  --pretrained-model google/flan-t5-large \
  --alert-phone "+18125841533" \
  --discord-webhook "https://discord.com/api/webhooks/1386109570283343953/uGkhj9dpuCg09SbKzZ0Tx2evugJrchQv-nrq3w0r_xi3w8si-XBpQJuxq_p_bcQlhB9W" \
  --resume-from-checkpoint /workspace/models/recipe_generation_flan-t5-large/checkpoint-1000

echo "✅ Training completed!"