import torch
import torch.distributed as dist
from smcfnet.core.launch import launch, get_rank, get_world_size

def test_worker():
  """Simple test worker to verify distributed setup."""
  rank = get_rank()
  world_size = get_world_size()
  
  print(f"[Rank {rank}/{world_size}] Worker initialized")
  
  # Test tensor communication
  if torch.cuda.is_available():
    device = torch.device(f'cuda:{rank}')
    tensor = torch.ones(3, 3).to(device) * rank
    print(f"[Rank {rank}] Created tensor on {device}:\n{tensor}")
    
    # Test all-reduce
    dist.all_reduce(tensor)
    print(f"[Rank {rank}] After all-reduce:\n{tensor}")
  
  print(f"[Rank {rank}] Test completed successfully!")

if __name__ == '__main__':
  print(f"PyTorch version: {torch.__version__}")
  print(f"CUDA available: {torch.cuda.is_available()}")
  print(f"CUDA version: {torch.version.cuda}")
  print(f"Number of GPUs: {torch.cuda.device_count()}")

  for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
      
  print(f"\nNCCL available: {torch.distributed.is_nccl_available()}")
  
  print("Starting distributed test...")
  launch(
      test_worker,
      num_gpus_per_machine=2,
      dist_url="auto",
  )
  print("Distributed test finished!")