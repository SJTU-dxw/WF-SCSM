from __future__ import annotations

import os
import torch
import torch.distributed as dist


def setup() -> tuple[torch.device, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    device = torch.device("cuda", local_rank)
    return device, rank, local_rank, world_size


def cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    result = value.detach().clone()
    dist.all_reduce(result)
    return result / dist.get_world_size()


def gather_with_local_gradient(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    gathered = [torch.zeros_like(value) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, value.detach())
    gathered[dist.get_rank()] = value
    return torch.cat(gathered)
