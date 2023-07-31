import os
import hostlist
import pickle as pkl
from pathlib import Path
import tempfile
import shutil

import torch
import torch.distributed as dist

import robolearn.utils.torch as ptu


def init_process(backend="nccl"):
    print(f"Starting process with rank {ptu.dist_rank}...", flush=True)

    gpu_ids = os.environ["SLURM_STEP_GPUS"].split(",")
    os.environ["MASTER_PORT"] = str(12345 + int(min(gpu_ids)))

    if "SLURM_JOB_NODELIST" in os.environ:
        hostnames = hostlist.expand_hostlist(os.environ["SLURM_JOB_NODELIST"])
        os.environ["MASTER_ADDR"] = hostnames[0]
    else:
        os.environ["MASTER_ADDR"] = "127.0.0.1"

    dist.init_process_group(
        backend,
        rank=ptu.dist_rank,
        world_size=ptu.world_size,
    )
    print(f"Process {ptu.dist_rank} is connected.", flush=True)
    dist.barrier()

    silence_print(ptu.dist_rank == 0)
    if ptu.dist_rank == 0:
        print(f"All processes are connected.", flush=True)


def silence_print(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def print_rank(s):
    print(f"rank {ptu.dist_rank}: {s}", flush=True)


def sync_model(sync_dir, model):
    # https://github.com/ylabbe/cosypose/blob/master/cosypose/utils/distributed.py
    sync_path = Path(sync_dir).resolve() / "sync_model.pkl"
    if ptu.dist_rank == 0 and ptu.world_size > 1:
        torch.save(model.state_dict(), sync_path)
    dist.barrier()
    if ptu.dist_rank > 0:
        model.load_state_dict(torch.load(sync_path))
    dist.barrier()
    if ptu.dist_rank == 0 and ptu.world_size > 1:
        sync_path.unlink()
    return model


def destroy_process():
    dist.destroy_process_group()


def barrier():
    dist.barrier()


def reduce_dict(input_dict, average=True):
    """
    https://github.com/pytorch/vision/blob/master/references/detection/utils.py
    Args:
        input_dict (dict): all the values will be reduced
        average (bool): whether to do average or sum
    Reduce the values in the dictionary from all processes so that all processes
    have the averaged results. Returns a dict with the same fields as
    input_dict, after reduction.
    """
    world_size = ptu.world_size
    if world_size < 2:
        return input_dict
    with torch.no_grad():
        names = []
        values = []
        reduced_dict = []
        # sort the keys so that they are consistent across processes
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
        values = torch.tensor(values).float().to(ptu.device)
        dist.all_reduce(values)
        if average:
            values /= world_size
        reduced_dict = {k: v.item() for k, v in zip(names, values)}
    return reduced_dict


def all_gather(data):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors)
    Args:
        data: any picklable object
    Returns:
        list[data]: list of data gathered from each rank
    """
    world_size = ptu.world_size
    if world_size == 1:
        return [data]

    # serialized to a Tensor
    buffer = pkl.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(ptu.device)

    # obtain Tensor size of each rank
    local_size = torch.tensor([tensor.numel()], device=ptu.device)
    size_list = [torch.tensor([0], device=ptu.device) for _ in range(world_size)]
    dist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # receiving Tensor from all ranks
    # we pad the tensor because torch all_gather does not support
    # gathering tensors of different shapes
    tensor_list = []
    for _ in size_list:
        tensor_list.append(
            torch.empty((max_size,), dtype=torch.uint8, device=ptu.device)
        )
    if local_size != max_size:
        padding = torch.empty(
            size=(max_size - local_size,),
            dtype=torch.uint8,
            device=ptu.device,
        )
        tensor = torch.cat((tensor, padding), dim=0)
    dist.all_gather(tensor_list, tensor)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pkl.loads(buffer))

    return data_list
