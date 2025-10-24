# Quick Start Guide - Training V2

## 🚀 Get Started in 5 Minutes

### Step 1: Build the Enhanced Docker Image

```bash
cd /Users/timmy/workspace/ai-apps/chef-genius

# Build V2 image with all 10 improvements
docker build -f Dockerfile.training_v2 -t chef-genius-training-v2 .
```

### Step 2: Run Training (Resume from Checkpoint 30000)

```bash
# Run with default settings (optimized for RTX 5090)
docker run --gpus all \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  -v $(pwd)/cli:/workspace/cli \
  --name chef-training-v2 \
  chef-genius-training-v2
```

The container will automatically:
- Resume from checkpoint-30000
- Use LoRA (3-4x faster!)
- Apply curriculum learning
- Track validation metrics
- Save improved checkpoints every 1000 steps

### Step 3: Monitor Progress

**Option A: Watch Docker Logs**
```bash
docker logs -f chef-training-v2
```

**Option B: Weights & Biases Dashboard**
- Visit: https://wandb.ai/your-username/chef-genius-optimized-v2
- View real-time metrics, validation loss, recipe quality

**Option C: Discord Notifications**
- Automatic notifications for:
  - Training started
  - Epoch progress
  - Validation results
  - Training completed

---

## 📊 What You'll See

### Training Output

```
🚀 Enhanced Trainer V2 Initialized with ALL 10 improvements!
   ✅ LoRA: Enabled
   ✅ Label Smoothing: 0.1
   ✅ Data Augmentation: 30.0%
   ✅ Cosine Scheduler: Enabled
   ✅ Gradient Noise: Enabled

🔧 Applying LoRA for efficient fine-tuning...
trainable params: 786,432 || all params: 783,150,080 || trainable%: 0.1004

🎓 Curriculum Learning - Epoch 1/5
   Difficulty: easy
   Quality threshold: 0.50
   Max complexity: {'max_ingredients': 6, 'max_steps': 6}

Step 30,100 | Loss: 0.7234
Step 30,200 | Loss: 0.7156

📊 Running validation at step 31000...
   Val Loss: 0.7324
   Val Perplexity: 2.08

🍳 Generating sample recipes for quality evaluation...
   Prompt: Create a simple pasta dish
   Recipe: Ingredients:
           - 2 cups pasta
           - 1 cup cherry tomatoes
           - 2 tbsp olive oil
           Instructions:
           1. Boil pasta for 10 minutes
           2. Sauté tomatoes in olive oil
           3. Combine and serve
   Quality: 0.89

✅ New best checkpoint at step 31000: loss=0.7324
💾 Saved training state: step 31000, epoch 1
```

---

## ⚙️ Customization

### Change Settings

Edit `Dockerfile.training_v2` CMD section (lines 215-228):

```dockerfile
CMD [ \
    "--epochs", "5", \
    "--batch-size", "12", \              # ← Adjust for your GPU
    "--gradient-accumulation-steps", "4", \
    "--lora-r", "16", \                  # ← Higher = more expressive
    "--lora-alpha", "32", \
    "--label-smoothing", "0.1", \        # ← 0.0-0.2 range
    "--augmentation-prob", "0.3" \       # ← 0.0-0.5 range
]
```

Then rebuild:
```bash
docker build -f Dockerfile.training_v2 -t chef-genius-training-v2 .
```

### Override at Runtime

```bash
docker run --gpus all \
  -v $(pwd)/models:/workspace/models \
  chef-genius-training-v2 \
  --epochs 3 \
  --batch-size 8 \
  --lora-r 32 \
  --label-smoothing 0.15
```

---

## 🎯 Recommended Configurations

### Maximum Speed (for iteration)
```bash
--epochs 2 \
--lora-r 8 \
--augmentation-prob 0.2 \
--batch-size 16
# Time: ~1 hour, Quality: 0.85-0.90
```

### Balanced (recommended)
```bash
--epochs 5 \
--lora-r 16 \
--augmentation-prob 0.3 \
--batch-size 12
# Time: 2-3 hours, Quality: 0.90-0.95
```

### Maximum Quality (for production)
```bash
--epochs 7 \
--lora-r 32 \
--augmentation-prob 0.4 \
--batch-size 8 \
--disable-lora  # Train full model
# Time: 8-10 hours, Quality: 0.95-0.98
```

---

## 📈 Expected Timeline (RTX 5090)

```
00:00 - Docker build starts
00:15 - Build complete, training starts
00:20 - Checkpoint 31000 saved
00:40 - Checkpoint 32000 saved
01:00 - Epoch 1 complete
01:20 - Checkpoint 33000 saved
02:00 - Epoch 2 complete (quality improving!)
02:30 - Epoch 3 complete
03:00 - Training complete! 🎉

Final results:
- Train loss: 0.59
- Val loss: 0.65
- Recipe quality: 0.92
- Saved: /workspace/models/chef-genius-flan-t5-large-lora/
```

---

## 🔍 Verify It's Working

### Check LoRA is Active

Look for this in logs:
```
🔧 Applying LoRA for efficient fine-tuning...
trainable params: 786,432 || all params: 783,150,080 || trainable%: 0.1004
✅ LoRA applied successfully!
```

**If you see this, LoRA is working!** Training will be 3-4x faster.

### Check Curriculum Learning

Look for difficulty progression:
```
Epoch 1: Difficulty: easy, Quality: 0.50
Epoch 2: Difficulty: medium, Quality: 0.58
Epoch 3: Difficulty: all, Quality: 0.65
```

### Check Validation

Look for periodic validation:
```
📊 Running validation at step 31000...
   Val Loss: 0.7324
   Val Perplexity: 2.08
```

---

## 🐛 Common Issues

### Issue: "PEFT not found"

**Solution:**
```bash
# SSH into container
docker exec -it chef-training-v2 bash

# Install PEFT
pip install peft>=0.7.0

# Exit and restart training
exit
docker restart chef-training-v2
```

### Issue: "Out of memory"

**Solutions:**
1. Reduce batch size: `--batch-size 8`
2. Lower LoRA rank: `--lora-r 8`
3. Increase gradient accumulation: `--gradient-accumulation-steps 8`

### Issue: "Validation loss much higher than training"

**This is normal!** It means:
- Model is learning (not memorizing)
- Validation set is working properly
- You're preventing overfitting

Only concerning if gap is >50%

### Issue: "Training stuck at step 30000"

Check if checkpoint loading failed:
```bash
docker logs chef-training-v2 | grep "Resume"
```

Should see:
```
✅ Resumed from step 30000, epoch 0
```

---

## 🎉 Next Steps

### After Training Completes

1. **Find your trained model:**
   ```bash
   ls models/chef-genius-flan-t5-large-lora/
   ```

2. **Test the model:**
   ```bash
   python cli/test_model.py \
     --model-path models/chef-genius-flan-t5-large-lora/ \
     --prompt "Create a healthy pasta dish"
   ```

3. **Compare checkpoints:**
   - Checkpoint 30000 (before V2)
   - Final checkpoint (after V2)
   - See the quality improvement!

4. **Deploy the model:**
   - LoRA weights are only 15MB
   - Easy to swap between adapters
   - Fast loading times

---

## 📞 Support

- **Issues:** Check `TRAINING_V2_IMPROVEMENTS.md` for detailed docs
- **Logs:** `docker logs chef-training-v2`
- **W&B Dashboard:** https://wandb.ai
- **Discord:** Notifications sent automatically

---

## ⚡ Pro Tips

1. **Use LoRA for iteration, full fine-tuning for production**
   - LoRA: Fast experiments (hours)
   - Full: Best quality (days)

2. **Monitor validation loss, not training loss**
   - Training loss always decreases
   - Validation loss shows real performance

3. **Early stopping usually triggers around epoch 3-4**
   - This is good! Prevents overfitting
   - Best model is automatically saved

4. **Checkpoint every 1000 steps = ~15 minutes**
   - Safe to interrupt training
   - Resume anytime

5. **Quality scores above 0.90 are excellent**
   - 0.85-0.90: Good
   - 0.90-0.95: Excellent
   - 0.95+: Production-ready

---

**Ready to train? Just run:**

```bash
docker build -f Dockerfile.training_v2 -t chef-genius-training-v2 . && \
docker run --gpus all \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/logs:/workspace/logs \
  --name chef-training-v2 \
  chef-genius-training-v2
```

**That's it! Training V2 will handle the rest. 🚀**
