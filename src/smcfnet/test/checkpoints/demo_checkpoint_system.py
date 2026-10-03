#!/usr/bin/env python
"""
Interactive demo of the checkpoint system.
Shows save/load workflow step by step.
"""
import sys
from pathlib import Path

print("=" * 70)
print("CHECKPOINT SYSTEM DEMO")
print("=" * 70)
print()
print("This demo will:")
print("  1. Train for 1 epoch and save a checkpoint")
print("  2. Inspect the saved checkpoint")
print("  3. Resume training from the checkpoint")
print()

# Check if we're ready to run
if not Path("src/smcfnet/config/checkpoint_demo.yaml").exists():
    print("❌ Config file not found. Please run from SMCM-MCFNet/ directory")
    sys.exit(1)

print("=" * 70)
print("STEP 1: Initial Training (1 epoch)")
print("=" * 70)
print()
print("Running: python -m smcfnet.core.train --config src/smcfnet/config/checkpoint_demo.yaml")
print()
print("Press Enter to continue...")
input()

import subprocess
result = subprocess.run(
    ["python", "-m", "smcfnet.core.train", "--config", "src/smcfnet/config/checkpoint_demo.yaml"],
    capture_output=False
)

if result.returncode != 0:
    print("\n❌ Training failed. Check errors above.")
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 2: Inspect the Checkpoint")
print("=" * 70)
print()
print("Checkpoint saved at: outputs/checkpoint_demo/epoch_1.pth")
print()
print("Press Enter to inspect it...")
input()

subprocess.run(["python", "inspect_checkpoint.py", "outputs/checkpoint_demo/epoch_1.pth"])

print("\n" + "=" * 70)
print("STEP 3: Resume Training")
print("=" * 70)
print()
print("Now we'll resume training from the saved checkpoint.")
print("The training will continue from epoch 1 (start of epoch 2).")
print()

# Create resume config
resume_config = """# Auto-generated resume config
dataset_config: "src/smcfnet/config/dsec.yaml"

data:
  event_bins: 5
  event_window_ms: 50
  image_height: 480
  image_width: 640
  num_workers: 2
  pin_memory: true

model:
  backbone: "cspdarknet"
  size: "nano"
  num_classes: 8
  pretrained: false
  event_representation: "voxel"
  mcm:
    enabled: false
  edum:
    enabled: false
  use_cmm: false
  fusion_layers: [3, 4, 5]

training:
  batch_size: 1
  num_epochs: 2  # Continue to epoch 2
  optimizer: "adam"
  learning_rate: 0.0001
  weight_decay: 0.0001
  lr_scheduler: "cosine"
  warmup_epochs: 0
  warmup_lr: 0.0001
  loss_obj: 1.0
  loss_cls: 1.0
  loss_box: 5.0
  grad_clip: 10.0
  use_amp: true
  use_mosaic: false
  use_mixup: false
  mosaic_prob: 0.0
  mixup_prob: 0.0

eval:
  eval_interval: 1
  save_interval: 1
  metrics: ["mAP50", "mAP"]
  iou_threshold: 0.5
  conf_threshold: 0.001
  nms_threshold: 0.65

log:
  output_dir: "./outputs"
  experiment_name: "checkpoint_demo_resumed"
  log_interval: 1
  save_all_checkpoints: true
  resume_from: "./outputs/checkpoint_demo/epoch_1.pth"  # RESUME FROM HERE
  load_weights_only: false  # Load full state

num_gpus: 1
dist_url: "auto"
seed: 42
"""

resume_config_path = Path("src/smcfnet/config/checkpoint_demo_resume.yaml")
resume_config_path.write_text(resume_config)

print(f"Created resume config at: {resume_config_path}")
print()
print("Press Enter to resume training...")
input()

result = subprocess.run(
    ["python", "-m", "smcfnet.core.train", "--config", str(resume_config_path)],
    capture_output=False
)

print("\n" + "=" * 70)
print("DEMO COMPLETE!")
print("=" * 70)
print()
print("Summary:")
print("  ✓ Trained for 1 epoch → saved checkpoint")
print("  ✓ Inspected checkpoint contents")
print("  ✓ Resumed training from checkpoint")
print()
print("Checkpoints created:")
print("  - outputs/checkpoint_demo/epoch_1.pth")
print("  - outputs/checkpoint_demo_resumed/epoch_2.pth")
print()
print("Try inspecting them:")
print("  python inspect_checkpoint.py outputs/checkpoint_demo/epoch_1.pth")
print("  python inspect_checkpoint.py outputs/checkpoint_demo_resumed/epoch_2.pth")
print()
