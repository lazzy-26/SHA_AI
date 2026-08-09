import os
from datetime import timedelta

import torch
import torch.distributed as dist


MASTER_ADDR = "172.20.10.2"
MASTER_PORT = 29501
WORLD_SIZE = 4

# All four PCs showed their wireless adapter as "WiFi".
os.environ["GLOO_SOCKET_IFNAME"] = "WiFi"
os.environ["USE_LIBUV"] = "0"


def main():
    rank_text = os.environ.get("RANK")

    if rank_text is None:
        raise RuntimeError(
            "RANK is not set. "
            "Set RANK=0, 1, 2, or 3 before running."
        )

    rank = int(rank_text)

    if rank < 0 or rank >= WORLD_SIZE:
        raise RuntimeError(
            f"Invalid rank {rank}. "
            f"Expected 0 to {WORLD_SIZE - 1}."
        )

    hostname = os.environ.get(
        "COMPUTERNAME",
        "unknown",
    )

    print("=" * 65)
    print("S.H.A DISTRIBUTED CONNECTION TEST")
    print("=" * 65)
    print(f"Computer    : {hostname}")
    print(f"Rank        : {rank}")
    print(f"World size  : {WORLD_SIZE}")
    print(f"Master      : {MASTER_ADDR}:{MASTER_PORT}")
    print(f"Gloo adapter: WiFi")
    print("=" * 65)

    #
    # IMPORTANT:
    #
    # Rank 0 listens on all IPv4 adapters.
    #
    # Ranks 1-3 connect specifically to the
    # hotspot address of Lazzy_PC.
    #
    store_host = (
        "0.0.0.0"
        if rank == 0
        else MASTER_ADDR
    )

    print(
        f"[Rank {rank}] Creating TCPStore "
        f"using {store_host}:{MASTER_PORT}...",
        flush=True,
    )

    store = dist.TCPStore(
        host_name=store_host,
        port=MASTER_PORT,
        world_size=WORLD_SIZE,
        is_master=(rank == 0),
        timeout=timedelta(seconds=180),
        wait_for_workers=True,
        use_libuv=False,
    )

    print(
        f"[Rank {rank}] TCPStore connected.",
        flush=True,
    )

    dist.init_process_group(
        backend="gloo",
        store=store,
        rank=rank,
        world_size=WORLD_SIZE,
        timeout=timedelta(seconds=180),
    )

    print(
        f"[Rank {rank}] Process group initialized.",
        flush=True,
    )

    value = torch.tensor(
        [float(rank + 1)],
        dtype=torch.float32,
    )

    print(
        f"[Rank {rank}] "
        f"Before all_reduce = {value.item()}",
        flush=True,
    )

    dist.all_reduce(
        value,
        op=dist.ReduceOp.SUM,
    )

    print(
        f"[Rank {rank}] "
        f"After all_reduce = {value.item()}",
        flush=True,
    )

    if value.item() != 10.0:
        raise RuntimeError(
            f"Distributed test failed. "
            f"Expected 10.0, got {value.item()}."
        )

    dist.barrier()

    if rank == 0:
        print()
        print("=" * 65)
        print(
            "S.H.A 4-NODE DISTRIBUTED CLUSTER TEST PASSED"
        )
        print("=" * 65)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()