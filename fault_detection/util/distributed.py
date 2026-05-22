import os
import sys
import torch
import torch.distributed as dist


def setup_distributed():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        return False, 0, 1

    torch.cuda.set_device(local_rank)

    if sys.platform == "win32":
        os.environ["USE_LIBUV"] = "0"
        backend = "gloo"
    else:
        backend = "nccl"

    dist.init_process_group(
        backend=backend, init_method="env://", rank=rank, world_size=world_size
    )
    return True, local_rank, world_size


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor):
    if not is_dist_avail_and_initialized():
        return tensor

    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt


def cleanup_distributed():
    if is_dist_avail_and_initialized():
        dist.destroy_process_group()
