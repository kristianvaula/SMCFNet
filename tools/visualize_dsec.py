"""
Visualization script for DSEC dataset samples.
Creates visualizations of RGB frames, event streams, and voxel grids.

Usage:
    python tools/visualize_dsec.py --sequence interlaken_00_e --frame 0
    python tools/visualize_dsec.py --sequence interlaken_00_e --frame 10 --output viz/
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root / 'src'))

from smcfnet.dataset.dsec.dsec_dataset import DSECDataset
from smcfnet.utils.viz_utils import (
    visualize_dsec_sample,
    visualize_event_stream,
    visualize_voxel_grid,
)
from smcfnet.utils.event_utils import events_to_voxel_grid



def main():
    parser = argparse.ArgumentParser(
        description="Visualize DSEC dataset samples"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="src/smcfnet/config/dsec.yaml",
        help="Path to DSEC config"
    )
    parser.add_argument(
        "--sequence",
        type=str,
        default=None,
        help="Sequence name (e.g., interlaken_00_e)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=-1,
        help="Frame index to visualize"
    )
    parser.add_argument(
        "--dt-ms",
        type=int,
        default=50,
        help="Event time window in milliseconds"
    )
    parser.add_argument(
        "--num-bins",
        type=int,
        default=5,
        help="Number of voxel bins"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="visualizations",
        help="Output directory for saved figures"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show plots interactively instead of saving"
    )
    args = parser.parse_args()
  
    
    # Create output directory
    output_dir = Path(args.output)
    if not args.show:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("DSEC Dataset Visualization")
    print("=" * 60)
    
    # Load dataset
    print(f"\nLoading dataset...")
    print(f"  Config: {args.config}")
    print(f"  Split: {args.split}")
    if args.sequence:
        print(f"  Sequence: {args.sequence}")
    
    try:
        ds = DSECDataset(
            config_path=args.config,
            sequence=args.sequence,
            split=args.split,
            dt_ms=args.dt_ms,
        )
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        sys.exit(1)
    
    print(f"\n✓ Dataset loaded successfully")
    print(f"  Sequence: {ds.sequence}")
    print(f"  Total frames: {len(ds)}")
    print(f"  Time window: {ds.dt_ms}ms")
    
    index = args.frame
    if index == -1:
      import random
      index = random.randint(0, min(len(ds)-1, 100))
    # Check frame index
    if index >= len(ds):
        print(f"\nError: Frame {index} out of range (max: {len(ds)-1})")
        sys.exit(1)
    
    # Load sample
    print(f"\nLoading frame {index}...")
    try:
        sample = ds[index]
    except Exception as e:
        print(f"Error loading frame: {e}")
        sys.exit(1)
    
    # Print sample info
    print(f"\n✓ Frame loaded successfully")
    print(f"  RGB shape: {sample['rgb'].shape if sample['rgb'] is not None else None}")
    print(f"  Number of events: {len(sample['events']['x'])}")
    print(f"  Time reference: {sample['t_ref']:.3f}s ({sample['t_ref_ms']}ms)")
    print(f"  Time window: {sample['dt']:.3f}s")
    
    # Get dimensions
    if sample['rgb'] is not None:
        _, height, width = sample['rgb'].shape
    else:
        # Default dimensions
        width, height = 640, 480
    
    # Create visualizations
    print(f"\nCreating visualizations...")
    
    # 1. RGB + Voxel Grid
    if sample['rgb'] is not None:
        voxel = events_to_voxel_grid(
            events=sample['events'],
            num_bins=args.num_bins,
            width=width,
            height=height,
        )
        
        save_path = None if args.show else output_dir / f"{ds.sequence}_frame{args.frame}_rgb_voxel.png"
        visualize_dsec_sample(sample, save_path, num_bins=args.num_bins)
        if save_path:
            print(f"  ✓ Saved RGB+Voxel to {save_path}")
    
    # 2. Event stream scatter plot
    if len(sample['events']['x']) > 0:
        save_path = None if args.show else output_dir / f"{ds.sequence}_frame{args.frame}_events.png"
        visualize_event_stream(
            events=sample['events'],
            width=width,
            height=height,
            save_path=save_path,
            title=f"Event Stream - {ds.sequence} Frame {args.frame}",
        )
        if save_path:
            print(f"  ✓ Saved event stream to {save_path}")
    
    # 3. Individual voxel bins
    if len(sample['events']['x']) > 0:
        voxel = events_to_voxel_grid(
            events=sample['events'],
            num_bins=args.num_bins,
            width=width,
            height=height,
        )
        
        save_path = None if args.show else output_dir / f"{ds.sequence}_frame{args.frame}_voxel_bins.png"
        visualize_voxel_grid(
            voxel=voxel,
            save_path=save_path,
            title=f"Voxel Grid - {ds.sequence} Frame {args.frame}",
        )
        if save_path:
            print(f"  ✓ Saved voxel bins to {save_path}")
    
    print("\n✓ Visualization complete!")
    
    if not args.show:
        print(f"\nVisualization saved to: {output_dir}/")
    
    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()