import os
from datetime import timedelta

import torch
import torch.distributed as dist


MASTER_ADDR = "172.20.10.2"
MASTER_PORT = 29400
WORLD_SIZE = 4


def main():
    rank_text = os.environ.get("RANK")

    if rank_text is None:
        raise RuntimeError(
            "RANK is not set. Set RANK=0, 1, 2 or 3 "
            "on each machine before running this script."
        )

    rank = int(rank_text)

    print(
        f"[Rank {rank}] Connecting to "
        f"{MASTER_ADDR}:{MASTER_PORT}...",
        flush=True,
    )

    dist.init_process_group(
        backend="gloo",
        init_method=(
            f"tcp://{MASTER_ADDR}:{MASTER_PORT}"
            "?use_libuv=0"
        ),
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=120),
    )

    hostname = os.environ.get(
        "COMPUTERNAME",
        "unknown",
    )

    value = torch.tensor(
        [float(rank + 1)],
        dtype=torch.float32,
    )

    print(
        f"[Rank {rank}] "
        f"Host={hostname} "
        f"Before={value.item()}",
        flush=True,
    )

    dist.all_reduce(
        value,
        op=dist.ReduceOp.SUM,
    )

    print(
        f"[Rank {rank}] "
        f"After all_reduce={value.item()}",
        flush=True,
    )

    dist.barrier()

    if rank == 0:
        print(
            "\nS.H.A 4-node distributed cluster test PASSED.",
            flush=True,
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()