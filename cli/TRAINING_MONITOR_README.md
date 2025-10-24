# Training Progress Monitor

This module provides tools to monitor and test your recipe generation model as it trains, allowing you to see sample outputs and evaluate progress without interrupting the training process.

## Features

- **Checkpoint Monitoring**: Automatically detects new training checkpoints and tests them
- **Sample Generation**: Generates sample recipes using various prompts to evaluate model quality
- **Device Detection**: Works with CUDA, MPS (Apple Silicon), and CPU devices
- **Flexible Testing**: Test individual checkpoints or monitor continuously

## Files

- `training_monitor.py` - Main monitoring script
- `train_with_monitoring_example.py` - Example of running training with parallel monitoring
- Updated `Makefile` with new monitoring commands

## Usage

### Quick Start with Makefile Commands

```bash
# List available datasets
make list-datasets

# Train model (saves to models/recipe_generation/)
make train-recipe-model

# Monitor training progress (run in separate terminal)
make monitor-training

# Test the current trained model
make test-model

# List available checkpoints
make list-checkpoints

# Test a specific checkpoint
make test-checkpoint CHECKPOINT=../models/recipe_generation/checkpoint-1000
```

### Direct Script Usage

#### Monitor Training Progress
```bash
# Monitor checkpoints as they're created (checks every 30 minutes)
cd cli
python training_monitor.py --monitor --model-dir ../models/recipe_generation

# Monitor with custom interval (e.g., every 10 minutes)
python training_monitor.py --monitor --model-dir ../models/recipe_generation --interval 10
```

#### Test Specific Checkpoints
```bash
# Test a specific checkpoint
python training_monitor.py --test-checkpoint ../models/recipe_generation/checkpoint-1000

# Test with custom prompts
python training_monitor.py --test-checkpoint ../models/recipe_generation/checkpoint-1000 \
    --prompts "<TITLE>Chocolate Cake" "<CUISINE>Thai" "<INGREDIENTS>beef | broccoli | soy sauce"
```

#### Test Current Model
```bash
# Test the final trained model
python training_monitor.py --model-dir ../models/recipe_generation

# With custom prompts and longer generation
python training_monitor.py --model-dir ../models/recipe_generation \
    --prompts "<TITLE>Vegetarian Pasta" "<CUISINE>Mexican" \
    --max-length 500
```

#### List Available Checkpoints
```bash
python training_monitor.py --list-checkpoints ../models/recipe_generation
```

### Training with Monitoring (Parallel)

Use the example script to run training and monitoring simultaneously:

```bash
cd cli

# Basic usage
python train_with_monitoring_example.py --model-output ../models/recipe_generation_test --epochs 5

# With custom parameters
python train_with_monitoring_example.py \
    --model-output ../models/recipe_generation_test \
    --epochs 10 \
    --batch-size 8 \
    --monitor-interval 15 \
    --datasets recipe_nlg indian_recipe_api

# Training only (no monitoring)
python train_with_monitoring_example.py \
    --model-output ../models/recipe_generation_test \
    --epochs 5 \
    --no-monitoring
```

## Default Test Prompts

The monitor uses these default prompts to test model quality:

- `<TITLE>Chocolate Chip Cookies`
- `<TITLE>Pasta Carbonara`
- `<TITLE>Chicken Curry`
- `<CUISINE>Italian`
- `<CUISINE>Mexican`
- `<INGREDIENTS>chicken breast | garlic | onion`

You can provide custom prompts using the `--prompts` argument.

## Output Format

Generated recipes follow the structured format used in training:

```
<TITLE>Recipe Name</TITLE>
<CUISINE>Cuisine Type</CUISINE>
<TIME>Cooking Time</TIME>
<SERVINGS>Number of Servings</SERVINGS>
<INGREDIENTS>ingredient1 | ingredient2 | ...</INGREDIENTS>
<INSTRUCTIONS>Step-by-step cooking instructions</INSTRUCTIONS>
```

## Tips for Effective Monitoring

1. **Run in Separate Terminal**: Start monitoring in a separate terminal window so you can see progress while training continues.

2. **Adjust Check Interval**: For long training runs, use longer intervals (30+ minutes). For shorter runs or debugging, use shorter intervals (5-10 minutes).

3. **Custom Prompts**: Test with prompts relevant to your use case. For example, if training on a specific cuisine, include prompts for that cuisine.

4. **Early Stopping**: Use the generated samples to decide if training is converging. If samples aren't improving after many checkpoints, you might want to stop training early.

5. **Device Compatibility**: The monitor automatically detects and uses the same device type as your training (CUDA, MPS, CPU).

## Troubleshooting

- **"Model files missing"**: This is normal for very early checkpoints. The monitor will retry as more checkpoints are created.
- **CUDA out of memory**: Reduce the `max_length` parameter or ensure no other processes are using GPU memory.
- **No checkpoints found**: Verify the output directory path and ensure training is actually creating checkpoints (check training logs).
- **Import errors**: Ensure you've installed the requirements: `pip install -r requirements.txt`

## Integration with Existing Workflow

This monitoring system is designed to work seamlessly with the existing training pipeline:

- Uses the same tokenizer and model architecture as `train_recipe_model.py`
- Compatible with all device types (CUDA, MPS, CPU)
- Follows the same structured recipe format
- Integrates with the existing Makefile workflow