"""
Utility script to inspect checkpoint files.
Shows contents, model structure, and training state.
"""
import sys
import argparse
import torch
from pathlib import Path
from collections import OrderedDict

def format_size(num_bytes):
    """Format bytes to human readable string."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"

def count_parameters(state_dict):
    """Count total parameters in model state dict."""
    total = 0
    for key, tensor in state_dict.items():
        if isinstance(tensor, torch.Tensor):
            total += tensor.numel()
    return total

def inspect_checkpoint(checkpoint_path):
    """Inspect a checkpoint file and print detailed information."""
    checkpoint_path = Path(checkpoint_path)
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return
    
    print("=" * 70)
    print(f"CHECKPOINT INSPECTION: {checkpoint_path.name}")
    print("=" * 70)
    
    # File size
    file_size = checkpoint_path.stat().st_size
    print(f"\n📁 File Info:")
    print(f"  Path: {checkpoint_path}")
    print(f"  Size: {format_size(file_size)}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    except Exception as e:
        print(f"\n❌ Error loading checkpoint: {e}")
        return
    
    print(f"\n✓ Checkpoint loaded successfully")
    
    # Top-level keys
    print(f"\n🔑 Top-level Keys:")
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], dict):
            print(f"  - {key}: dict with {len(checkpoint[key])} entries")
        elif isinstance(checkpoint[key], (int, float, str)):
            print(f"  - {key}: {checkpoint[key]}")
        else:
            print(f"  - {key}: {type(checkpoint[key]).__name__}")
    
    # Training state
    print(f"\n📊 Training State:")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'iteration' in checkpoint:
        print(f"  Iteration: {checkpoint['iteration']}")
    if 'best_metric' in checkpoint:
        print(f"  Best Metric: {checkpoint['best_metric']:.4f}")
    
    # Model info
    if 'model' in checkpoint:
        model_state = checkpoint['model']
        num_params = count_parameters(model_state)
        print(f"\n🧠 Model:")
        print(f"  Total Parameters: {num_params:,}")
        print(f"  State Dict Keys: {len(model_state)}")
        
        # Show some layer names (first 10 and last 10)
        keys = list(model_state.keys())
        print(f"\n  First 5 layers:")
        for key in keys[:5]:
            shape = model_state[key].shape if isinstance(model_state[key], torch.Tensor) else "N/A"
            print(f"    - {key}: {shape}")
        
        if len(keys) > 10:
            print(f"    ... ({len(keys) - 10} more layers) ...")
        
        print(f"\n  Last 5 layers:")
        for key in keys[-5:]:
            shape = model_state[key].shape if isinstance(model_state[key], torch.Tensor) else "N/A"
            print(f"    - {key}: {shape}")
    
    # Optimizer info
    if 'optimizer' in checkpoint:
        opt_state = checkpoint['optimizer']
        print(f"\n⚙️  Optimizer:")
        if 'param_groups' in opt_state:
            print(f"  Parameter Groups: {len(opt_state['param_groups'])}")
            if opt_state['param_groups']:
                pg = opt_state['param_groups'][0]
                print(f"  Learning Rate: {pg.get('lr', 'N/A')}")
                print(f"  Weight Decay: {pg.get('weight_decay', 'N/A')}")
        if 'state' in opt_state:
            print(f"  State Entries: {len(opt_state['state'])}")
    
    # Scheduler info
    if 'scheduler' in checkpoint:
        sched_state = checkpoint['scheduler']
        print(f"\n📈 Scheduler:")
        for key, value in sched_state.items():
            if not key.startswith('_'):
                print(f"  {key}: {value}")
    
    # AMP scaler
    if 'scaler' in checkpoint:
        print(f"\n⚡ AMP Scaler: Present (Mixed precision training)")
    
    # Config
    if 'config' in checkpoint:
        print(f"\n⚙️  Config: Saved with checkpoint")
        config = checkpoint['config']
        if hasattr(config, 'model'):
            print(f"  Model Size: {config.model.size}")
            print(f"  Num Classes: {config.model.num_classes}")
        if hasattr(config, 'training'):
            print(f"  Batch Size: {config.training.batch_size}")
            print(f"  Learning Rate: {config.training.learning_rate}")
    
    print("\n" + "=" * 70)

def compare_checkpoints(ckpt1_path, ckpt2_path):
    """Compare two checkpoints."""
    print("=" * 70)
    print("CHECKPOINT COMPARISON")
    print("=" * 70)
    
    ckpt1 = torch.load(ckpt1_path, map_location='cpu')
    ckpt2 = torch.load(ckpt2_path, map_location='cpu')
    
    print(f"\nCheckpoint 1: {Path(ckpt1_path).name}")
    print(f"  Epoch: {ckpt1.get('epoch', 'N/A')}")
    print(f"  Best Metric: {ckpt1.get('best_metric', 'N/A')}")
    
    print(f"\nCheckpoint 2: {Path(ckpt2_path).name}")
    print(f"  Epoch: {ckpt2.get('epoch', 'N/A')}")
    print(f"  Best Metric: {ckpt2.get('best_metric', 'N/A')}")
    
    # Compare model keys
    keys1 = set(ckpt1['model'].keys()) if 'model' in ckpt1 else set()
    keys2 = set(ckpt2['model'].keys()) if 'model' in ckpt2 else set()
    
    if keys1 == keys2:
        print(f"\n✓ Model architectures match ({len(keys1)} layers)")
    else:
        print(f"\n⚠️  Model architectures differ")
        only_in_1 = keys1 - keys2
        only_in_2 = keys2 - keys1
        if only_in_1:
            print(f"  Only in checkpoint 1: {len(only_in_1)} layers")
        if only_in_2:
            print(f"  Only in checkpoint 2: {len(only_in_2)} layers")
    
    print("\n" + "=" * 70)

def list_checkpoints(directory):
    """List all checkpoints in a directory."""
    directory = Path(directory)
    
    if not directory.exists():
        print(f"❌ Directory not found: {directory}")
        return
    
    # Find all .pth files
    checkpoints = sorted(directory.glob("**/*.pth"))
    
    if not checkpoints:
        print(f"No checkpoints found in {directory}")
        return
    
    print("=" * 70)
    print(f"CHECKPOINTS IN: {directory}")
    print("=" * 70)
    
    for ckpt_path in checkpoints:
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu')
            epoch = ckpt.get('epoch', '?')
            best = ckpt.get('best_metric', '?')
            size = format_size(ckpt_path.stat().st_size)
            
            rel_path = ckpt_path.relative_to(directory)
            print(f"\n📦 {rel_path}")
            print(f"   Epoch: {epoch} | Best: {best} | Size: {size}")
        except Exception as e:
            print(f"\n❌ {ckpt_path.name}: Error loading - {e}")
    
    print("\n" + "=" * 70)

def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint files")
    parser.add_argument("checkpoint", nargs='?', help="Path to checkpoint file")
    parser.add_argument("--compare", help="Compare with another checkpoint")
    parser.add_argument("--list", help="List all checkpoints in directory")
    
    args = parser.parse_args()
    
    if args.list:
        list_checkpoints(args.list)
    elif args.checkpoint:
        inspect_checkpoint(args.checkpoint)
        if args.compare:
            compare_checkpoints(args.checkpoint, args.compare)
    else:
        # Default: look for checkpoints in outputs
        outputs_dir = Path("outputs")
        if outputs_dir.exists():
            list_checkpoints(outputs_dir)
        else:
            parser.print_help()

if __name__ == '__main__':
    main()
