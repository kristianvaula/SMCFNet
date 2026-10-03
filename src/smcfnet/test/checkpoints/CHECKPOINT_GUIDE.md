# Checkpoint System Guide

This guide explains how to save, load, and manage checkpoints in the SMCM-MCFNet training system.

## Overview

The checkpoint system allows you to:
- Save training progress at regular intervals
- Resume interrupted training from where you left off
- Load pre-trained weights for fine-tuning or inference
- Track the best performing model during training

## Checkpoint Contents

Each checkpoint file (`.pth`) contains:
- **Training State**: epoch, iteration, best metric
- **Model Weights**: complete model parameter state
- **Optimizer State**: momentum, learning rate, etc.
- **Scheduler State**: learning rate schedule progress
- **AMP Scaler** (if using mixed precision)
- **Configuration**: the config used for training

## Configuration Options

In your config file (e.g., `debug.yaml`):

```yaml
log:
  output_dir: "./outputs"              # Base directory for outputs
  experiment_name: "my_experiment"     # Subdirectory for this run
  resume_from: null                    # Path to checkpoint to resume from
  load_weights_only: false             # If true, only load model weights

eval:
  save_interval: 1                     # Save checkpoint every N epochs
```

## Usage Examples

### 1. Automatic Checkpointing During Training

Checkpoints are saved automatically during training:

```bash
# Start training - saves checkpoints every epoch (based on save_interval)
python -m smcfnet.core.train --config src/smcfnet/config/debug.yaml
```

This creates:
- `outputs/debug_test/epoch_1.pth`
- `outputs/debug_test/epoch_2.pth`
- `outputs/debug_test/best.pth` (when validation improves)
- etc.

### 2. Resume Training from Checkpoint

To continue training from a saved checkpoint:

```yaml
# In your config file:
log:
  resume_from: "./outputs/debug_test/epoch_2.pth"
  load_weights_only: false  # Load full training state
```

Or via command line:
```bash
python -m smcfnet.core.train \
  --config src/smcfnet/config/debug.yaml \
  --resume "./outputs/debug_test/epoch_2.pth"
```

**What happens:**
- Training continues from saved epoch
- Optimizer state is restored
- Learning rate schedule continues correctly
- Best metric tracking continues

### 3. Load Weights Only (Fine-tuning)

To start fresh training with pre-trained weights:

```yaml
# In your config file:
log:
  resume_from: "./outputs/baseline/best.pth"
  load_weights_only: true  # Only load model weights
```

**What happens:**
- Only model parameters are loaded
- Training starts from epoch 0
- Optimizer is freshly initialized
- Useful for transfer learning or fine-tuning

### 4. Load for Inference

```python
from smcfnet.core.train import build_model
from smcfnet.config.config import get_config

config, _ = get_config()
model = build_model(config)

# Load checkpoint
checkpoint = torch.load("outputs/debug_test/best.pth")
model.load_state_dict(checkpoint['model'])
model.eval()

# Ready for inference
```

## Testing the Checkpoint System

### Run Comprehensive Tests

```bash
python test_checkpoint.py
```

This tests:
1. ✓ Checkpoint saving
2. ✓ Full state restoration
3. ✓ Weights-only loading
4. ✓ Device compatibility
5. ✓ State dict completeness

### Inspect Checkpoint Files

```bash
# Inspect a specific checkpoint
python inspect_checkpoint.py outputs/debug_test/epoch_1.pth

# List all checkpoints in a directory
python inspect_checkpoint.py --list outputs/

# Compare two checkpoints
python inspect_checkpoint.py \
  outputs/debug_test/epoch_1.pth \
  --compare outputs/debug_test/epoch_2.pth
```

## Best Practices

### 1. Checkpoint Management

```yaml
eval:
  save_interval: 5              # Save every 5 epochs (not every epoch)
```

The `PeriodicCheckpointer` hook automatically:
- Keeps only the last 5 checkpoints (configurable)
- Always preserves `best.pth`
- Deletes older checkpoints to save disk space

### 2. Long Training Runs

For multi-day training:

```yaml
log:
  resume_from: "./outputs/experiment/epoch_50.pth"  # Resume if interrupted
  save_all_checkpoints: false                       # Save disk space
  
eval:
  save_interval: 10  # Save less frequently for long runs
```

### 3. Distributed Training

Checkpoints are only saved by the main process (rank 0):
- Other processes skip checkpoint operations
- Prevents file conflicts in multi-GPU training

### 4. Storage Considerations

Checkpoint file sizes:
- **nano**: ~2-5 MB
- **tiny**: ~5-10 MB
- **small**: ~10-20 MB
- **medium**: ~30-50 MB
- **large**: ~50-100 MB

Plan disk space accordingly for long training runs.

## Troubleshooting

### Checkpoint Not Found

```
Error: Checkpoint not found: ./outputs/experiment/epoch_10.pth
```

**Solution**: Check the path is correct relative to where you run the command.

### Model Architecture Mismatch

```
Error: size mismatch for backbone.stem.conv.weight
```

**Solution**: The checkpoint was saved with a different model size. Verify:
```bash
python inspect_checkpoint.py outputs/experiment/epoch_10.pth
```

### CUDA Out of Memory on Resume

If you get OOM when resuming:
1. Check if you're loading on the correct device
2. Ensure batch size matches the original training
3. Clear CUDA cache: `torch.cuda.empty_cache()`

### Weights Not Loading Properly

```python
# Always use map_location when loading across devices
checkpoint = torch.load(ckpt_path, map_location='cpu')
# or
checkpoint = torch.load(ckpt_path, map_location=device)
```

## Advanced: Manual Checkpoint Handling

For custom checkpoint logic:

```python
from smcfnet.core.trainer import Trainer

# Save custom checkpoint
trainer.save_checkpoint("my_checkpoint.pth")

# Load with custom logic
checkpoint = torch.load("my_checkpoint.pth", map_location=device)
trainer.load_state_dict(checkpoint)

# Or just weights
trainer.model.load_state_dict(checkpoint['model'])
```

## Checkpoint Workflow Diagram

```
Training Start
     │
     ├─> Epoch 1 ─────────> epoch_1.pth
     │
     ├─> Epoch 2 ─────────> epoch_2.pth
     │                       (if eval improves: best.pth)
     ├─> Epoch 3 ─────────> epoch_3.pth
     │
     └─> [Interrupted] ──────┐
                             │
         Resume Training ────┘
              │
              ├─> Epoch 4 ─> epoch_4.pth
              └─> ...
```

## Quick Reference

| Task | Config Setting |
|------|----------------|
| Save every N epochs | `eval.save_interval: N` |
| Resume training | `log.resume_from: "path/to/checkpoint.pth"` |
| Load weights only | `log.load_weights_only: true` |
| Change save location | `log.output_dir: "new/path"` |
| Experiment naming | `log.experiment_name: "my_exp"` |

## Additional Resources

- See [trainer.py](src/smcfnet/core/trainer.py) for implementation details
- See [hooks.py](src/smcfnet/core/hooks.py) for checkpoint hook logic
- Run `python test_checkpoint.py` for comprehensive tests
