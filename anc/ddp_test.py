import os

import torch
import torch.distributed as dist


def main():
    dist.init_process_group(
        backend="gloo",
        init_method="env://",
    )

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    hostname = os.environ.get(
        "COMPUTERNAME",
        "unknown",
    )

    value = torch.tensor(
        [float(rank + 1)]
    )

    print(
        f"[Rank {rank}] "
        f"Host={hostname} "
        f"World={world_size} "
        f"Before={value.item()}",
        flush=True,
    )

    dist.all_reduce(
        value,
        op=dist.ReduceOp.SUM,
    )

    print(
        f"[Rank {rank}] "
        f"After all_reduce="
        f"{value.item()}",
        flush=True,
    )

    dist.barrier()

    if rank == 0:
        print(
            "S.H.A distributed cluster "
            "test PASSED.",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()