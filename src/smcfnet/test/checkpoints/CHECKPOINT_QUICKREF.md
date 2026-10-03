# Checkpoint System - Quick Reference Card

## 🎯 Core Commands

```bash
# Test the system
python test_checkpoint.py

# Inspect a checkpoint
python inspect_checkpoint.py path/to/checkpoint.pth

# List all checkpoints
python inspect_checkpoint.py --list outputs/

# Compare checkpoints
python inspect_checkpoint.py ckpt1.pth --compare ckpt2.pth

# Interactive demo
python demo_checkpoint_system.py
```

## 📝 Configuration

```yaml
log:
  resume_from: "./outputs/exp/epoch_5.pth"  # null = start fresh
  load_weights_only: false                  # true = weights only, false = full state

eval:
  save_interval: 5  # Save every N epochs
```

## 🔄 Common Use Cases

### 1. Resume Interrupted Training
```yaml
log:
  resume_from: "./outputs/experiment/epoch_10.pth"
  load_weights_only: false  # Restore full state
```
- Continues from saved epoch
- Optimizer state restored
- LR schedule continues correctly

### 2. Fine-tune from Pretrained
```yaml
log:
  resume_from: "./outputs/baseline/best.pth"
  load_weights_only: true  # Only model weights
```
- Loads pretrained weights
- Fresh optimizer/scheduler
- Starts from epoch 0

### 3. Load for Inference
```python
checkpoint = torch.load("best.pth", map_location='cpu')
model.load_state_dict(checkpoint['model'])
model.eval()
```

## 📦 Checkpoint Contents

- `epoch` - Training epoch number
- `iteration` - Global iteration count
- `best_metric` - Best validation metric
- `model` - Model parameters (690 keys for nano)
- `optimizer` - Optimizer state
- `scheduler` - LR scheduler state
- `scaler` - AMP scaler (if using mixed precision)
- `config` - Training configuration

## 💾 File Sizes (Nano Model)

- Checkpoint: ~13-14 MB
- Contains full training state
- Device-agnostic (CPU ↔ CUDA)

## ✅ Verification

All tests passing ✓
- Checkpoint saving
- Full state restoration
- Weights-only loading
- Cross-device compatibility
- State dict completeness

## 📚 Documentation

- **CHECKPOINT_GUIDE.md** - Complete guide with examples
- **CHECKPOINT_TESTING.md** - System status and test results
- **test_checkpoint.py** - Comprehensive test suite
- **inspect_checkpoint.py** - Inspection utility
- **demo_checkpoint_system.py** - Interactive demo

## 🐛 Quick Troubleshooting

| Problem | Solution |
|---------|----------|
| "Checkpoint not found" | Check path relative to pwd |
| "Size mismatch" | Wrong model size in checkpoint |
| CUDA OOM | Match batch size to original training |
| "KeyError" | Checkpoint from old code version |

## 🚀 Quick Start

```bash
# 1. Run tests
python test_checkpoint.py

# 2. Train and save
python -m smcfnet.core.train --config src/smcfnet/config/debug.yaml

# 3. Inspect result
python inspect_checkpoint.py outputs/debug_test/epoch_1.pth

# 4. Resume training
# Edit config: resume_from: "./outputs/debug_test/epoch_1.pth"
python -m smcfnet.core.train --config src/smcfnet/config/resume_test.yaml
```

## 💡 Best Practices

✓ Save every 5-10 epochs (not every epoch) for long runs
✓ Keep `best.pth` separate (automatic)
✓ Use `weights_only=true` for transfer learning
✓ Always use `map_location` when loading
✓ Verify checkpoint before long training runs

## 🎓 Training Workflow

```
Start Training
    ↓
Every N epochs → Save checkpoint
    ↓
Eval improves? → Save best.pth
    ↓
[Interrupted?]
    ↓
Resume from checkpoint → Continue training
```
