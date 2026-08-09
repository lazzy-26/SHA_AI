import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from config import (
    BATCH_SIZE,
    BEST_MODEL_PATH,
    CLEAN_DIR,
    EPOCHS,
    LAST_MODEL_PATH,
    LEARNING_RATE,
    NOISE_DIR,
    NUM_WORKERS,
    SEED,
    TRAIN_SPLIT,
    VAL_SPLIT,
)
from dataset import ANCDataset, find_wav_files
from model import ANCNet


def setup_distributed():
    """
    Initialize distributed training.

    torchrun automatically provides:
        RANK
        LOCAL_RANK
        WORLD_SIZE
        MASTER_ADDR
        MASTER_PORT
    """

    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available in this PyTorch build."
        )

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    # Windows distributed training uses Gloo.
    backend = "gloo"

    dist.init_process_group(
        backend=backend,
        init_method="env://",
    )

    return rank, local_rank, world_size


def cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return (
        not dist.is_initialized()
        or dist.get_rank() == 0
    )


def print_main(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)


def set_seed(seed, rank):
    """
    Give every worker a deterministic but different random seed.
    """

    worker_seed = seed + rank

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(worker_seed)


def split_files(files):
    """
    Produce the exact same train/validation/test file split
    on every machine.
    """

    files = list(files)

    rng = random.Random(SEED)
    rng.shuffle(files)

    total = len(files)

    train_end = int(
        total * TRAIN_SPLIT
    )

    val_end = train_end + int(
        total * VAL_SPLIT
    )

    train_files = files[:train_end]

    validation_files = files[
        train_end:val_end
    ]

    test_files = files[val_end:]

    # Small-dataset protection.
    if not train_files and files:
        train_files = files[:1]

    if not validation_files and train_files:
        validation_files = train_files[:1]

    return (
        train_files,
        validation_files,
        test_files,
    )


def choose_device(local_rank):
    """
    Use one GPU per worker when CUDA is available.

    On CPU-only machines, use CPU + Gloo.
    """

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()

        if device_count == 0:
            return torch.device("cpu")

        device_index = local_rank % device_count

        torch.cuda.set_device(
            device_index
        )

        return torch.device(
            f"cuda:{device_index}"
        )

    return torch.device("cpu")


def build_loader(
    dataset,
    rank,
    world_size,
    shuffle,
):
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        seed=SEED,
        drop_last=False,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )

    return loader, sampler


def reduce_mean(value, device):
    """
    Average a scalar across all distributed workers.
    """

    tensor = torch.tensor(
        value,
        dtype=torch.float64,
        device=device,
    )

    dist.all_reduce(
        tensor,
        op=dist.ReduceOp.SUM,
    )

    tensor /= dist.get_world_size()

    return tensor.item()


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
):
    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_batches = 0

    context = (
        torch.enable_grad()
        if training
        else torch.no_grad()
    )

    with context:
        for noisy, clean in loader:
            noisy = noisy.to(
                device,
                non_blocking=True,
            )

            clean = clean.to(
                device,
                non_blocking=True,
            )

            if training:
                optimizer.zero_grad(
                    set_to_none=True
                )

            enhanced = model(noisy)

            loss = criterion(
                enhanced,
                clean,
            )

            if training:
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=5.0,
                )

                optimizer.step()

            total_loss += loss.item()
            total_batches += 1

    local_average = (
        total_loss /
        max(total_batches, 1)
    )

    global_average = reduce_mean(
        local_average,
        device,
    )

    return global_average


def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    validation_loss,
    world_size,
):
    """
    Only rank 0 calls this.

    model.module is the real ANCNet underneath DDP.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_model = (
        model.module
        if isinstance(model, DDP)
        else model
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict":
                raw_model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "validation_loss":
                validation_loss,
            "world_size":
                world_size,
        },
        path,
    )


def main():
    rank = 0
    local_rank = 0
    world_size = 1

    try:
        (
            rank,
            local_rank,
            world_size,
        ) = setup_distributed()

        set_seed(
            SEED,
            rank,
        )

        device = choose_device(
            local_rank
        )

        clean_files = find_wav_files(
            CLEAN_DIR
        )

        noise_files = find_wav_files(
            NOISE_DIR
        )

        if is_main_process():
            print("=" * 72)
            print(
                "S.H.A ANC V1 DISTRIBUTED TRAINING"
            )
            print("=" * 72)

            print(
                f"Workers      : {world_size}"
            )

            print(
                f"Clean files  : "
                f"{len(clean_files)}"
            )

            print(
                f"Noise files  : "
                f"{len(noise_files)}"
            )

        if len(clean_files) < 3:
            raise RuntimeError(
                "At least 3 clean WAV files are required."
            )

        if not noise_files:
            raise RuntimeError(
                "At least 1 noise WAV file is required."
            )

        (
            train_clean,
            validation_clean,
            test_clean,
        ) = split_files(
            clean_files
        )

        train_dataset = ANCDataset(
            train_clean,
            noise_files,
            length_multiplier=10,
        )

        validation_dataset = ANCDataset(
            validation_clean,
            noise_files,
            length_multiplier=3,
        )

        (
            train_loader,
            train_sampler,
        ) = build_loader(
            train_dataset,
            rank,
            world_size,
            shuffle=True,
        )

        (
            validation_loader,
            validation_sampler,
        ) = build_loader(
            validation_dataset,
            rank,
            world_size,
            shuffle=False,
        )

        if is_main_process():
            print(
                f"Train clean  : "
                f"{len(train_clean)}"
            )

            print(
                f"Validation   : "
                f"{len(validation_clean)}"
            )

            print(
                f"Test clean   : "
                f"{len(test_clean)}"
            )

            print(
                f"Batch/worker : "
                f"{BATCH_SIZE}"
            )

            print(
                f"Global batch : "
                f"{BATCH_SIZE * world_size}"
            )

            print(
                f"Epochs       : "
                f"{EPOCHS}"
            )

            print(
                f"Learning rate: "
                f"{LEARNING_RATE}"
            )

            print("=" * 72)

        print(
            f"[Rank {rank}] "
            f"device={device} "
            f"local_rank={local_rank}"
        )

        model = ANCNet().to(
            device
        )

        if device.type == "cuda":
            model = DDP(
                model,
                device_ids=[
                    device.index
                ],
                output_device=
                    device.index,
            )
        else:
            model = DDP(model)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
        )

        criterion = nn.L1Loss()

        best_validation_loss = float(
            "inf"
        )

        # Make sure every worker has
        # finished initialization.
        dist.barrier()

        for epoch in range(
            1,
            EPOCHS + 1,
        ):
            # Required for proper distributed
            # shuffling between epochs.
            train_sampler.set_epoch(
                epoch
            )

            validation_sampler.set_epoch(
                epoch
            )

            train_loss = run_epoch(
                model=model,
                loader=train_loader,
                criterion=criterion,
                device=device,
                optimizer=optimizer,
            )

            validation_loss = run_epoch(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=device,
            )

            if is_main_process():
                print(
                    f"Epoch "
                    f"{epoch:03d}/"
                    f"{EPOCHS} | "
                    f"train="
                    f"{train_loss:.6f} | "
                    f"val="
                    f"{validation_loss:.6f}"
                )

                save_checkpoint(
                    path=LAST_MODEL_PATH,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    validation_loss=
                        validation_loss,
                    world_size=
                        world_size,
                )

                if (
                    validation_loss
                    < best_validation_loss
                ):
                    best_validation_loss = (
                        validation_loss
                    )

                    save_checkpoint(
                        path=
                            BEST_MODEL_PATH,
                        model=model,
                        optimizer=
                            optimizer,
                        epoch=epoch,
                        validation_loss=
                            validation_loss,
                        world_size=
                            world_size,
                    )

                    print(
                        "  -> New best "
                        "checkpoint saved"
                    )

            # Every worker waits until
            # rank 0 finishes saving.
            dist.barrier()

        if is_main_process():
            print()
            print("=" * 72)
            print(
                "DISTRIBUTED TRAINING COMPLETE"
            )
            print("=" * 72)

            print(
                "Best validation loss: "
                f"{best_validation_loss:.6f}"
            )

            print(
                f"Best checkpoint: "
                f"{BEST_MODEL_PATH}"
            )

            print(
                f"Last checkpoint: "
                f"{LAST_MODEL_PATH}"
            )

    except Exception as error:
        print(
            f"[Rank {rank}] ERROR: "
            f"{error}",
            flush=True,
        )

        raise

    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()