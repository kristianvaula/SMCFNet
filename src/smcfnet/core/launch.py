"""
Multi-GPU training launcher adapted from detectron2.
Handles distributed training setup and process spawning.

Based on https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py
"""
import logging
import os
import socket
from datetime import timedelta
from typing import Callable, Tuple, Any

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = timedelta(minutes=30)


def _get_env_int(name: str, default: int) -> int:
  val = os.environ.get(name)
  if val is None:
    return default
  try:
    return int(val)
  except ValueError:
    return default

def _find_free_port() -> int:
  """Find an available port on localhost."""
  sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  sock.bind(("", 0))
  port = sock.getsockname()[1]
  sock.close()
  return port

def launch (
  main_func: Callable,
  num_gpus_per_machine: int,
  num_machines: int = 1,
  machine_rank: int = 0,
  dist_url: str = None,
  args: Tuple[Any, ...] = (),
  timeout: timedelta = DEFAULT_TIMEOUT,
) -> None:
  """
  Launch multi-process or distributed training.
  
  Args: 
    main_func: Function to be called as main_func(*args)
    num_gpus_per_machine: Number of processes per machine (GPUs)
    num_machines: Total number of machines
    machine_rank: Rank of this machine
    dist_url: URL for distributed jobs (e.g., "tcp://127.0.0.1:8686" or "auto")
    args: Arguments passed to main_func
    timeout: Timeout for distributed workers
  """
  world_size = num_machines * num_gpus_per_machine
  
  if world_size == 1:
    main_func(*args)
    return

  # Default distributed URL.
  # - env:// works with torchrun and is robust across backends.
  # - For our mp.spawn single-machine case, we ensure MASTER_ADDR/PORT are set.
  if dist_url is None:
    dist_url = "env://"
  
  if dist_url == "env://" and num_machines == 1:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_find_free_port()))

  if dist_url == "auto":
    assert num_machines == 1, "dist_url=auto not supported in multi-machine jobs"
    port = _find_free_port()
    dist_url = f"tcp://127.0.0.1:{port}"
    
  if num_machines > 1 and dist_url and dist_url.startswith("file://"):
    logger.warning("file:// is not reliable for multi-machine jobs. Prefer tcp://")

  os.environ.setdefault("WORLD_SIZE", str(world_size))
  os.environ.setdefault("LOCAL_WORLD_SIZE", str(num_gpus_per_machine))
  
  mp.spawn(
    _distributed_worker,
    nprocs=num_gpus_per_machine,
    args=(
      main_func,
      world_size,
      num_gpus_per_machine,
      machine_rank,
      dist_url,
      args,
      timeout,
    ),
    daemon=False,
  )

def _distributed_worker(
  local_rank: int,
  main_func: Callable,
  world_size: int,
  num_gpus_per_machine: int,
  machine_rank: int,
  dist_url: str,
  args: Tuple[Any, ...],
  timeout: timedelta = DEFAULT_TIMEOUT,
) -> None:
  """Worker function for distributed training"""
  has_gpu = torch.cuda.is_available()
  if has_gpu:
    assert num_gpus_per_machine <= torch.cuda.device_count(), f"Param {num_gpus_per_machine} higher than cuda device count"
    torch.cuda.set_device(local_rank)
  global_rank = machine_rank * num_gpus_per_machine + local_rank
  
  # Populate standard env vars expected by torch.distributed + common launchers.
  os.environ["RANK"] = str(global_rank)
  os.environ["WORLD_SIZE"] = str(world_size)
  os.environ["LOCAL_RANK"] = str(local_rank)
  os.environ["LOCAL_WORLD_SIZE"] = str(num_gpus_per_machine)
  
  try:
    dist.init_process_group(
      backend="NCCL" if has_gpu else "GLOO",
      init_method=dist_url,
      world_size=world_size,
      rank=global_rank,
      timeout=timeout,
    )
  except Exception as e:
    logger.error(f"Process group URL: {dist_url}")
    raise e
  
  # Synchronize to prevent timeout issues
  if has_gpu:
    dist.barrier(device_ids=[local_rank])
  else:
    dist.barrier()
  
  try:
    main_func(*args)
  finally:
    # Clean up process group to avoid warnings
    dist.destroy_process_group()


def get_rank() -> int:
  """Get the rank of current process."""
  if not dist.is_available() or not dist.is_initialized():
    # torchrun sets RANK/WORLD_SIZE env vars even if the process group is not
    # initialized (e.g. for simple multi-process scripts/tests).
    return _get_env_int("RANK", 0)
  return int(dist.get_rank())


def get_local_rank() -> int:
  """Get local rank (GPU index) for current process.

  Prefers the standard LOCAL_RANK environment variable (set by torchrun
  or by our mp.spawn launcher). Falls back to 0.
  """
  return _get_env_int("LOCAL_RANK", 0)


def get_world_size() -> int:
  """Get total number of processes."""
  if not dist.is_available():
    return int(_get_env_int("WORLD_SIZE", 1))
  if dist.is_initialized():
    return int(dist.get_world_size())
  return int(_get_env_int("WORLD_SIZE", 1))


def is_main_process() -> bool:
  """Check if current process is main process."""
  return get_rank() == 0


def synchronize() -> None:
  """Synchronize all processes."""
  if get_world_size() == 1:
    return
  if (
    torch.cuda.is_available()
    and dist.is_available()
    and dist.is_initialized()
    and dist.get_backend().lower() == "nccl"
  ):
    dist.barrier(device_ids=[get_local_rank()])
  else:
    dist.barrier()